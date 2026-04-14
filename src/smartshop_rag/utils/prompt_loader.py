from smartshop_rag.utils.config_handler import prompts_conf
from smartshop_rag.utils.logger_handler import logger
from smartshop_rag.utils.path_tool import get_abs_path


def load_system_prompts() -> str:
    return _load_prompt("main_prompt_path", "主提示词")


def load_rag_prompts() -> str:
    return _load_prompt("rag_summarize_prompt_path", "RAG 提示词")


def load_query_rewrite_prompt() -> str:
    return _load_prompt("query_rewrite_prompt_path", "Query Rewrite 提示词")


def load_rerank_prompt() -> str:
    return _load_prompt("rerank_prompt_path", "Rerank 提示词")


def _load_prompt(config_key: str, label: str) -> str:
    try:
        prompt_path = get_abs_path(prompts_conf[config_key])
    except KeyError as exc:
        logger.error(f"[加载提示词]未找到配置项: {config_key}")
        raise exc

    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as exc:
        logger.error(f"[加载提示词]解析{label}失败: {str(exc)}")
        raise exc
