import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from smartshop_rag.utils.config_handler import chroma_conf
from smartshop_rag.utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, md_loader, pdf_loader, txt_loader
from smartshop_rag.utils.path_tool import get_abs_path


MANIFEST_SCHEMA_VERSION = 1
MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass
class ChunkManifest:
    meta: dict[str, Any]
    documents: list[Document]


@dataclass
class MarkdownSection:
    content: str
    heading_path: str
    heading_level: int
    section_index: int


@dataclass
class MarkdownSectionGroup:
    sections: list[MarkdownSection]

    @property
    def content(self) -> str:
        return "\n\n".join(section.content for section in self.sections if section.content).strip()

    @property
    def heading_paths(self) -> list[str]:
        return [section.heading_path for section in self.sections]

    @property
    def heading_path(self) -> str:
        return " | ".join(dict.fromkeys(path for path in self.heading_paths if path))

    @property
    def heading_level(self) -> int:
        if not self.sections:
            return 0
        return min(section.heading_level for section in self.sections)

    @property
    def section_index_start(self) -> int:
        return self.sections[0].section_index

    @property
    def section_index_end(self) -> int:
        return self.sections[-1].section_index


def get_chunk_manifest_path(manifest_path: str | None = None) -> Path:
    target = manifest_path or chroma_conf["chunk_manifest_path"]
    return Path(get_abs_path(target))


def chunk_manifest_exists(manifest_path: str | None = None) -> bool:
    return get_chunk_manifest_path(manifest_path).exists()


def ensure_chunk_manifest_ready(manifest_path: str | None = None) -> None:
    path = get_chunk_manifest_path(manifest_path)
    if path.exists():
        return
    raise FileNotFoundError(f"chunk manifest 不存在: {path}. 请先运行 `python src/smartshop_rag/rag/ingest.py` 完成知识库构建。")


def _is_supported_knowledge_file(file_path: Path) -> bool:
    normalized = str(file_path).replace("/", "\\").lower()
    return "\\cleaned\\" in normalized or "\\shared\\policies\\" in normalized


def get_knowledge_source_files(data_path: str | None = None) -> list[str]:
    target_data_path = get_abs_path(data_path or chroma_conf["data_path"])
    if not Path(target_data_path).is_dir():
        raise FileNotFoundError(f"知识库目录不存在: {target_data_path}")
    allowed_files = listdir_with_allowed_type(target_data_path, tuple(chroma_conf["allow_knowledge_file_type"]))
    return [path for path in allowed_files if _is_supported_knowledge_file(Path(path))]


def parse_knowledge_metadata(path: str) -> dict[str, str]:
    normalized = Path(path)
    parts = normalized.parts
    brand = "unknown"
    model = "shared"
    doc_type = normalized.stem
    for index, part in enumerate(parts):
        if part == "midea":
            brand = part
            if index + 1 < len(parts):
                next_part = parts[index + 1]
                model = "shared" if next_part == "shared" else next_part
            break
    source_path = normalized.as_posix()
    metadata = {
        "brand": brand,
        "model": model,
        "doc_type": doc_type,
        "source_path": source_path,
        "source": source_path,
    }
    if model == "shared":
        metadata["shared"] = "true"
    return metadata


def load_file_documents(read_path: str) -> list[Document]:
    lower_path = read_path.lower()
    if lower_path.endswith(".txt"):
        return txt_loader(read_path)
    if lower_path.endswith(".pdf"):
        return pdf_loader(read_path)
    if lower_path.endswith(".md"):
        return md_loader(read_path)
    return []


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chroma_conf["chunk_size"],
        chunk_overlap=chroma_conf["chunk_overlap"],
        separators=chroma_conf["separators"],
        length_function=len,
    )


def get_chunk_strategy() -> dict[str, Any]:
    return {
        "chunk_strategy_version": int(chroma_conf.get("chunk_strategy_version", 1)),
        "boundary_strategy": "markdown_semantic_merged",
        "merge_short_sections": True,
        "chunk_size": int(chroma_conf["chunk_size"]),
        "chunk_overlap": int(chroma_conf["chunk_overlap"]),
        "separators": list(chroma_conf["separators"]),
        "allowed_file_types": list(chroma_conf["allow_knowledge_file_type"]),
    }


def build_chunked_documents_for_file(path: str, *, splitter: RecursiveCharacterTextSplitter | None = None, source_md5: str | None = None) -> list[Document]:
    documents = load_file_documents(path)
    if not documents:
        return []
    text_splitter = splitter or create_text_splitter()
    base_metadata = parse_knowledge_metadata(path)
    if source_md5:
        base_metadata["source_md5"] = source_md5
    if path.lower().endswith(".md"):
        chunked_documents = build_markdown_semantic_chunks(documents, base_metadata=base_metadata, splitter=text_splitter)
    else:
        for doc in documents:
            doc.metadata = {**doc.metadata, **base_metadata}
        chunked_documents = text_splitter.split_documents(documents)

    for index, doc in enumerate(chunked_documents):
        metadata = {**base_metadata, **doc.metadata}
        metadata["chunk_index"] = str(index)
        metadata["chunk_id"] = f"{base_metadata['source_path']}#chunk_{index}"
        doc.metadata = metadata
    return chunked_documents


def build_markdown_semantic_chunks(
    documents: list[Document],
    *,
    base_metadata: dict[str, str],
    splitter: RecursiveCharacterTextSplitter,
) -> list[Document]:
    chunked_documents: list[Document] = []
    for doc in documents:
        sections = parse_markdown_sections(doc.page_content or "")
        for group in group_markdown_sections(sections, max_chars=int(chroma_conf["chunk_size"])):
            group_content = group.content
            section_metadata = {
                **doc.metadata,
                **base_metadata,
                "heading_path": group.heading_path,
                "heading_level": str(group.heading_level),
                "section_index": str(group.section_index_start),
                "section_index_start": str(group.section_index_start),
                "section_index_end": str(group.section_index_end),
                "merged_section_count": str(len(group.sections)),
                "heading_path_list": json.dumps(group.heading_paths, ensure_ascii=False),
            }
            if len(group_content) <= int(chroma_conf["chunk_size"]):
                chunked_documents.append(
                    Document(
                        page_content=group_content,
                        metadata={**section_metadata, "chunk_in_section_index": "0"},
                    )
                )
                continue

            split_documents = splitter.split_documents([Document(page_content=group_content, metadata=section_metadata)])
            for chunk_index, split_doc in enumerate(split_documents):
                split_doc.metadata = {
                    **section_metadata,
                    **split_doc.metadata,
                    "chunk_in_section_index": str(chunk_index),
                }
                chunked_documents.append(split_doc)
    return chunked_documents


def group_markdown_sections(sections: list[MarkdownSection], *, max_chars: int) -> list[MarkdownSectionGroup]:
    groups: list[MarkdownSectionGroup] = []
    current: list[MarkdownSection] = []

    def flush_current() -> None:
        nonlocal current
        if current:
            groups.append(MarkdownSectionGroup(sections=current))
            current = []

    for section in sections:
        if _is_query_hint_section(section):
            flush_current()
            groups.append(MarkdownSectionGroup(sections=[section]))
            continue

        if len(section.content) > max_chars:
            flush_current()
            groups.append(MarkdownSectionGroup(sections=[section]))
            continue

        if not current:
            current = [section]
            continue

        candidate = [*current, section]
        if _can_merge_markdown_sections(current[-1], section) and _merged_content_length(candidate) <= max_chars:
            current = candidate
            continue

        flush_current()
        current = [section]

    flush_current()
    return groups


def _merged_content_length(sections: list[MarkdownSection]) -> int:
    return len("\n\n".join(section.content for section in sections if section.content).strip())


def _is_query_hint_section(section: MarkdownSection) -> bool:
    return "典型问法映射" in section.heading_path


def _can_merge_markdown_sections(previous: MarkdownSection, current: MarkdownSection) -> bool:
    if _is_query_hint_section(previous) or _is_query_hint_section(current):
        return False
    return _merge_group_key(previous.heading_path) == _merge_group_key(current.heading_path)


def _merge_group_key(heading_path: str) -> str:
    parts = [part.strip() for part in heading_path.split(" > ") if part.strip()]
    if not parts:
        return ""
    if parts[0] == "商品详情":
        return parts[0]
    if parts[0] == "使用说明" and len(parts) >= 2:
        if parts[1] in {"快速入门", "清洁保养"}:
            return "使用说明 > 操作保养"
        return " > ".join(parts[:2])
    return parts[0]


def parse_markdown_sections(text: str) -> list[MarkdownSection]:
    sections: list[MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_heading_path = ""
    current_heading_level = 0
    current_has_body = False

    def flush_current() -> None:
        nonlocal current_lines, current_heading_path, current_heading_level, current_has_body
        content = "\n".join(current_lines).strip()
        if content and current_has_body:
            sections.append(
                MarkdownSection(
                    content=content,
                    heading_path=current_heading_path,
                    heading_level=current_heading_level,
                    section_index=len(sections),
                )
            )
        current_lines = []
        current_heading_path = ""
        current_heading_level = 0
        current_has_body = False

    for line in text.splitlines():
        heading_match = MARKDOWN_HEADING_PATTERN.match(line)
        if heading_match:
            flush_current()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip().strip("#").strip()
            heading_stack = [(existing_level, existing_title) for existing_level, existing_title in heading_stack if existing_level < level]
            heading_stack.append((level, title))
            current_heading_path = " > ".join(existing_title for _, existing_title in heading_stack)
            current_heading_level = level
            current_lines = [line]
            current_has_body = False
            continue

        current_lines.append(line)
        if line.strip():
            current_has_body = True

    flush_current()
    return sections


def build_current_chunk_manifest(data_path: str | None = None) -> ChunkManifest:
    splitter = create_text_splitter()
    source_files: list[dict[str, str]] = []
    documents: list[Document] = []
    for path in get_knowledge_source_files(data_path):
        source_md5 = get_file_md5_hex(path)
        if not source_md5:
            raise RuntimeError(f"知识文件 MD5 计算失败: {path}")
        source_files.append({"source_path": Path(path).as_posix(), "md5": source_md5})
        documents.extend(build_chunked_documents_for_file(path, splitter=splitter, source_md5=source_md5))

    meta_without_fingerprint = {
        "record_type": "manifest_meta",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "chunk_strategy": get_chunk_strategy(),
        "source_files": source_files,
        "source_file_count": len(source_files),
        "chunk_count": len(documents),
    }
    fingerprint = calculate_manifest_fingerprint(meta_without_fingerprint)
    meta = {**meta_without_fingerprint, "manifest_fingerprint": fingerprint}
    return ChunkManifest(meta=meta, documents=documents)


def calculate_manifest_fingerprint(meta_without_fingerprint: dict[str, Any]) -> str:
    payload = json.dumps(meta_without_fingerprint, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_chunk_manifest(manifest: ChunkManifest, manifest_path: str | None = None) -> Path:
    path = get_chunk_manifest_path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(manifest.meta, ensure_ascii=False, sort_keys=True) + "\n")
        for doc in manifest.documents:
            metadata = dict(doc.metadata)
            record = {
                "record_type": "chunk",
                "chunk_id": str(metadata.get("chunk_id", "")),
                "page_content": doc.page_content,
                "metadata": metadata,
                "source_path": str(metadata.get("source_path", "")),
                "source_md5": str(metadata.get("source_md5", "")),
                "chunk_index": str(metadata.get("chunk_index", "")),
            }
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_chunk_manifest(manifest_path: str | None = None) -> ChunkManifest:
    path = get_chunk_manifest_path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"chunk manifest 不存在: {path}. 请先运行 `python src/smartshop_rag/rag/ingest.py` 完成知识库构建。")

    with path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    if not lines:
        raise ValueError(f"chunk manifest 为空: {path}")

    meta = json.loads(lines[0])
    if meta.get("record_type") != "manifest_meta":
        raise ValueError(f"chunk manifest 首行不是 manifest_meta: {path}")
    documents: list[Document] = []
    for line in lines[1:]:
        record = json.loads(line)
        if record.get("record_type") != "chunk":
            continue
        metadata = dict(record.get("metadata") or {})
        if record.get("chunk_id") and not metadata.get("chunk_id"):
            metadata["chunk_id"] = str(record["chunk_id"])
        documents.append(Document(page_content=str(record.get("page_content", "")), metadata=metadata))
    return ChunkManifest(meta=meta, documents=documents)


def read_manifest_fingerprint(manifest_path: str | None = None) -> str | None:
    path = get_chunk_manifest_path(manifest_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        first_line = file.readline().strip()
    if not first_line:
        return None
    try:
        meta = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    fingerprint = meta.get("manifest_fingerprint")
    return str(fingerprint) if fingerprint else None
