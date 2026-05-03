import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smartshop_rag.rag.vector_store import VectorStoreService, get_knowledge_source_files



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="构建 SmartShop-RAG 本地 chunk manifest 与 Chroma 向量库")
    parser.add_argument("--reset", action="store_true", help="清空旧 Chroma、MD5 记录和 chunk manifest 后重建")
    args = parser.parse_args(argv or [])

    files = get_knowledge_source_files()
    print(f"发现 {len(files)} 个可处理知识文件。")

    service = VectorStoreService()
    stats = service.load_document(reset=args.reset)
    print(
        "建库完成: "
        f"扫描 {stats['scanned']} 个文件, "
        f"新增 {stats['loaded']} 个文件, "
        f"跳过 {stats['skipped']} 个文件, "
        f"失败 {stats['failed']} 个文件。"
    )
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

