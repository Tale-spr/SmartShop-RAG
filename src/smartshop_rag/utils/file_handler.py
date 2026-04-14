import hashlib
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from smartshop_rag.utils.logger_handler import logger


def get_file_md5_hex(filepath: str):
    if not os.path.exists(filepath):
        logger.error(f"[md5计算]文件{filepath}不存在")
        return None
    if not os.path.isfile(filepath):
        logger.error(f"[md5计算]路径{filepath}不是文件")
        return None

    md5_obj = hashlib.md5()
    try:
        with open(filepath, "rb") as file:
            while chunk := file.read(4096):
                md5_obj.update(chunk)
        return md5_obj.hexdigest()
    except Exception as exc:
        logger.error(f"[md5计算]文件{filepath}md5计算失败, {str(exc)}")
        return None


def listdir_with_allowed_type(path: str, allowed_types: tuple[str]):
    root = Path(path)
    if not root.is_dir():
        logger.error(f"[文件列表]路径{path}不是文件夹")
        return tuple()

    normalized_types = tuple(s.lower() if s.startswith('.') else f'.{s.lower()}' for s in allowed_types)
    files = [str(file_path) for file_path in root.rglob('*') if file_path.is_file() and file_path.suffix.lower() in normalized_types]
    return tuple(sorted(files))


def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    return PyPDFLoader(filepath, passwd).load()


def txt_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()


def md_loader(filepath: str) -> list[Document]:
    return TextLoader(filepath, encoding="utf-8").load()
