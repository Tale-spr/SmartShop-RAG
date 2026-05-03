import shutil
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from smartshop_rag.model.factory import create_embedding_model
from smartshop_rag.rag.chunk_manifest import (
    build_current_chunk_manifest,
    chunk_manifest_exists,
    create_text_splitter,
    get_chunk_manifest_path,
    get_knowledge_source_files as _get_knowledge_source_files,
    load_file_documents,
    parse_knowledge_metadata,
    read_chunk_manifest,
    read_manifest_fingerprint,
    write_chunk_manifest,
)
from smartshop_rag.utils.config_handler import chroma_conf
from smartshop_rag.utils.logger_handler import logger
from smartshop_rag.utils.path_tool import get_abs_path, get_project_root

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


def ensure_chunk_manifest_ready(manifest_path: str | None = None) -> None:
    path = get_chunk_manifest_path(manifest_path)
    if path.exists():
        return
    raise FileNotFoundError(
        f"chunk manifest 不存在: {path}. 请先运行 `python src/smartshop_rag/rag/ingest.py` 完成知识库构建。"
    )


def get_knowledge_source_files(data_path: str | None = None) -> list[str]:
    return _get_knowledge_source_files(data_path)


def _parse_knowledge_metadata(path: str) -> dict[str, str]:
    return parse_knowledge_metadata(path)


def _load_file_documents(read_path: str) -> list[Document]:
    return load_file_documents(read_path)


def _ensure_generated_path(path: Path, *, label: str) -> Path:
    project_root = Path(get_project_root()).resolve()
    resolved = path.resolve(strict=False)
    if resolved == project_root or project_root not in resolved.parents:
        raise ValueError(f"{label} 不在项目目录内，拒绝清理: {resolved}")
    return resolved


def clear_generated_index_storage(*, remove_manifest: bool = True) -> None:
    vector_store_dir = _ensure_generated_path(get_vector_store_directory(), label="向量库目录")
    if vector_store_dir.exists():
        shutil.rmtree(vector_store_dir)

    md5_path = _ensure_generated_path(Path(get_abs_path(chroma_conf["md5_hex_store"])), label="MD5 记录文件")
    if md5_path.exists():
        md5_path.unlink()

    if remove_manifest:
        manifest_path = _ensure_generated_path(get_chunk_manifest_path(), label="chunk manifest")
        if manifest_path.exists():
            manifest_path.unlink()
        if manifest_path.parent.exists() and not any(manifest_path.parent.iterdir()):
            manifest_path.parent.rmdir()


class VectorStoreService:
    def __init__(self, embedding_function: Embeddings | None = None):
        self.embedding_function = embedding_function or create_embedding_model()
        self._vector_store: Chroma | None = None
        self.spliter = create_text_splitter()

    @property
    def vector_store(self) -> Chroma:
        return self._get_vector_store()

    def _get_vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=chroma_conf["collection_name"],
                embedding_function=self.embedding_function,
                persist_directory=str(get_vector_store_directory()),
            )
        return self._vector_store

    def get_retriever(self, k: int | None = None):
        return self._get_vector_store().as_retriever(search_kwargs={"k": k or chroma_conf["k"]})

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
        if data_path is not None:
            logger.warning("[加载知识库]load_all_chunked_documents 已改为从 manifest 读取, data_path 参数将被忽略")
        return read_chunk_manifest().documents

    def vector_search(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        docs = self._get_vector_store().similarity_search(query, k=top_k)
        return [
            {
                "document": doc,
                "score": None,
                "rank": rank,
                "source": "vector",
            }
            for rank, doc in enumerate(docs, start=1)
        ]

    def load_document(self, *, reset: bool = False) -> dict[str, int]:
        stats = {"scanned": 0, "loaded": 0, "skipped": 0, "failed": 0}
        try:
            current_manifest = build_current_chunk_manifest()
            stats["scanned"] = int(current_manifest.meta.get("source_file_count", 0))
            current_fingerprint = str(current_manifest.meta["manifest_fingerprint"])
            existing_fingerprint = read_manifest_fingerprint()
            if not reset and chunk_manifest_exists() and vector_store_exists() and existing_fingerprint == current_fingerprint:
                stats["skipped"] = stats["scanned"]
                logger.info("[加载知识库]chunk manifest 与向量库已是最新,跳过重建")
                return stats

            clear_generated_index_storage(remove_manifest=True)
            write_chunk_manifest(current_manifest)
            self._vector_store = None
            documents = current_manifest.documents
            if documents:
                ids = [str(doc.metadata.get("chunk_id")) for doc in documents]
                self._get_vector_store().add_documents(documents, ids=ids)
                stats["loaded"] = stats["scanned"]
                logger.info(f"[加载知识库]重建完成: {len(documents)} 个 chunk 已写入向量库")
            else:
                stats["skipped"] = stats["scanned"]
                logger.warning("[加载知识库]没有有效 chunk 写入向量库")
        except Exception as exc:
            stats["failed"] = max(stats["failed"], 1)
            logger.error(f"[加载知识库]重建失败: {str(exc)}", exc_info=True)
        return stats
