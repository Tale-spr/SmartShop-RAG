from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from smartshop_rag.model.responses_chat import QwenResponsesChatModel
from smartshop_rag.utils.config_handler import rag_conf

CHAT_MODEL_ROLE_MAP = {
    "primary_chat": "primary_chat",
    "rag_chat": "rag_chat",
    "rewrite_chat": "rewrite_chat",
    "rerank_chat": "rerank_chat",
    "eval_chat": "eval_chat",
}


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def __init__(self, model_name: str | None = None, role: str = "primary_chat"):
        self.model_name = model_name
        self.role = role

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        model_name = self.model_name or get_chat_model_name(self.role)
        if is_responses_api_model(model_name):
            return QwenResponsesChatModel(model=model_name)
        return ChatTongyi(model=model_name)


class EmbeddingFactory(BaseModelFactory):
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=self.model_name or get_embedding_model_name())


def _get_models_config() -> dict[str, str]:
    models = rag_conf.get("models")
    if not isinstance(models, dict):
        raise ValueError("config/rag.yml 缺少 models 配置")
    return models


def get_chat_model_name(role: str = "primary_chat") -> str:
    normalized_role = CHAT_MODEL_ROLE_MAP.get(role)
    if normalized_role is None:
        raise ValueError(f"未知聊天模型角色: {role}")

    models = _get_models_config()
    model_name = models.get(normalized_role)
    if not model_name:
        raise ValueError(f"config/rag.yml 中未配置 models.{normalized_role}")
    return str(model_name)


def is_responses_api_model(model_name: str) -> bool:
    normalized = model_name.strip().lower()
    return normalized in {
        "qwen3.5-plus",
        "qwen3.5-plus-2026-02-15",
        "qwen3.6-plus",
        "qwen3.6-plus-2026-04-02",
    }


def get_embedding_model_name() -> str:
    models = _get_models_config()
    model_name = models.get("embedding")
    if not model_name:
        raise ValueError("config/rag.yml 中未配置 models.embedding")
    return str(model_name)


def create_chat_model(model_name: str | None = None, role: str = "primary_chat") -> BaseChatModel:
    model = ChatModelFactory(model_name=model_name, role=role).generator()
    if model is None:
        raise ValueError("聊天模型初始化失败")
    return model


def create_embedding_model(model_name: str | None = None) -> Embeddings:
    model = EmbeddingFactory(model_name=model_name).generator()
    if model is None:
        raise ValueError("Embedding 模型初始化失败")
    return model
