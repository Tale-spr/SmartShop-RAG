import os
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from smartshop_rag.model.factory import create_embedding_model
from smartshop_rag.utils.config_handler import chroma_conf
from smartshop_rag.utils.file_handler import get_file_md5_hex, listdir_with_allowed_type, md_loader, pdf_loader, txt_loader
from smartshop_rag.utils.logger_handler import logger
from smartshop_rag.utils.path_tool import get_abs_path

VECTOR_STORE_SQLITE = "chroma.sqlite3"


def get_vector_store_directory(persist_directory: str | None = None) -> Path:
    target = persist_directory or chroma_conf["persist_directory"]
    return Path(get_abs_path(target))


def get_vector_store_sqlite_path(persist_directory: str | None = None) -> Path:
    return get_vector_store_directory(persist_directory) / VECTOR_STORE_SQLITE


def vector_store_exists(persist_directory: str | None = None) -> bool:
    return get_vector_store_sqlite_path(persist_directory).exists()


def ensure_vector_store_ready(persist_directory: str | None = None) -> None:
    sqlite_path = get_vector_store_sqlite_path(persist_directory)
    if sqlite_path.exists():
        return
    raise FileNotFoundError(
        f"本地向量库不存在: {sqlite_path}. 请先运行 `python src/smartshop_rag/rag/ingest.py` 完成知识库构建。"
    )


def _is_supported_knowledge_file(file_path: Path) -> bool:
    normalized = str(file_path).replace("/", "\\").lower()
    return "\\cleaned\\" in normalized or "\\shared\\policies\\" in normalized


def get_knowledge_source_files(data_path: str | None = None) -> list[str]:
    target_data_path = get_abs_path(data_path or chroma_conf["data_path"])
    if not os.path.isdir(target_data_path):
        raise FileNotFoundError(f"知识库目录不存在: {target_data_path}")
    allowed_files = listdir_with_allowed_type(target_data_path, tuple(chroma_conf["allow_knowledge_file_type"]))
    return [path for path in allowed_files if _is_supported_knowledge_file(Path(path))]


def _parse_knowledge_metadata(path: str) -> dict[str, str]:
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


def _load_file_documents(read_path: str) -> list[Document]:
    lower_path = read_path.lower()
    if lower_path.endswith(".txt"):
        return txt_loader(read_path)
    if lower_path.endswith(".pdf"):
        return pdf_loader(read_path)
    if lower_path.endswith(".md"):
        return md_loader(read_path)
    return []


class VectorStoreService:
    def __init__(self, embedding_function: Embeddings | None = None):
        self.embedding_function = embedding_function or create_embedding_model()
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=self.embedding_function,
            persist_directory=chroma_conf["persist_directory"],
        )
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self, k: int | None = None):
        return self.vector_store.as_retriever(search_kwargs={"k": k or chroma_conf["k"]})

    def get_chunked_documents(self, path: str) -> list[Document]:
        documents = _load_file_documents(path)
        if not documents:
            return []
        base_metadata = _parse_knowledge_metadata(path)
        for doc in documents:
            doc.metadata = {**doc.metadata, **base_metadata}
        split_documents = self.spliter.split_documents(documents)
        for index, doc in enumerate(split_documents):
            metadata = {**base_metadata, **doc.metadata}
            metadata["chunk_index"] = str(index)
            metadata["chunk_id"] = f"{base_metadata['source_path']}#chunk_{index}"
            doc.metadata = metadata
        return split_documents

    def load_all_chunked_documents(self, data_path: str | None = None) -> list[Document]:
        all_documents: list[Document] = []
        for path in get_knowledge_source_files(data_path):
            all_documents.extend(self.get_chunked_documents(path))
        return all_documents

    def vector_search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        docs = self.vector_store.similarity_search(query, k=top_k)
        return [
            {
                "document": doc,
                "score": None,
                "rank": rank,
                "source": "vector",
            }
            for rank, doc in enumerate(docs, start=1)
        ]

    def load_document(self) -> dict[str, int]:
        stats = {"scanned": 0, "loaded": 0, "skipped": 0, "failed": 0}
        md5_store_path = get_abs_path(chroma_conf["md5_hex_store"])

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(md5_store_path):
                open(md5_store_path, "w", encoding="utf-8").close()
                return False
            with open(md5_store_path, "r", encoding="utf-8") as file:
                return any(line.strip() == md5_for_check for line in file.readlines())

        def save_md5_hex(md5_for_save: str):
            with open(md5_store_path, "a", encoding="utf-8") as file:
                file.write(md5_for_save + "\n")

        for path in get_knowledge_source_files():
            stats["scanned"] += 1
            md5_hex = get_file_md5_hex(path)
            if not md5_hex:
                stats["failed"] += 1
                logger.error(f"[加载知识库]{path}MD5计算失败")
                continue
            if check_md5_hex(md5_hex):
                stats["skipped"] += 1
                logger.info(f"[加载知识库]{path}内容已经存在知识库内,跳过")
                continue
            try:
                split_document = self.get_chunked_documents(path)
                if not split_document:
                    stats["skipped"] += 1
                    logger.warning(f"[加载知识库]{path}没有有效文本内容,跳过")
                    continue
                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                stats["loaded"] += 1
                logger.info(f"[加载知识库]{path}内容加载成功")
            except Exception as exc:
                stats["failed"] += 1
                logger.error(f"[加载知识库]{path}加载失败: {str(exc)}", exc_info=True)
        return stats
