from __future__ import annotations

import os
from typing import Any

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class QwenResponsesChatModel(BaseChatModel):
    model: str
    api_key: str | None = None
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    timeout: int = 120
    enable_thinking: bool | None = None

    @property
    def _llm_type(self) -> str:
        return "qwen_responses_api"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "backend": "responses_api",
        }

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = run_manager
        payload: dict[str, Any] = {
            "model": self.model,
            "input": self._serialize_messages(messages),
        }
        if stop:
            payload["stop"] = stop
        if self.enable_thinking is not None:
            payload["enable_thinking"] = self.enable_thinking
        if kwargs:
            payload.update(kwargs)

        response = requests.post(
            f"{self.base_url.rstrip('/')}/responses",
            headers={
                "Authorization": f"Bearer {self._get_api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text.strip()
            raise ValueError(
                f"Responses API 调用失败: model={self.model}, backend=responses_api, "
                f"status={response.status_code}, body={body}"
            ) from exc

        data = response.json()
        content = self._extract_output_text(data)
        generation = ChatGeneration(
            message=AIMessage(content=content),
            generation_info={
                "response_id": data.get("id"),
                "backend": "responses_api",
            },
        )
        return ChatResult(
            generations=[generation],
            llm_output={
                "model_name": self.model,
                "backend": "responses_api",
                "response_id": data.get("id"),
                "token_usage": data.get("usage", {}),
            },
        )

    def _get_api_key(self) -> str:
        api_key = self.api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY，无法调用 Responses API")
        return api_key

    @staticmethod
    def _serialize_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
        serialized: list[dict[str, str]] = []
        for message in messages:
            role = QwenResponsesChatModel._message_role(message)
            content = QwenResponsesChatModel._stringify_content(message.content)
            if not content:
                continue
            serialized.append({"role": role, "content": content})
        return serialized

    @staticmethod
    def _message_role(message: BaseMessage) -> str:
        if isinstance(message, SystemMessage):
            return "system"
        if isinstance(message, HumanMessage):
            return "user"
        if isinstance(message, AIMessage):
            return "assistant"
        if isinstance(message, ChatMessage):
            return message.role
        return "user"

    @staticmethod
    def _stringify_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    chunks.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunk for chunk in chunks if chunk).strip()
        return str(content).strip()

    @staticmethod
    def _extract_output_text(data: dict[str, Any]) -> str:
        output_text = data.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output_items = data.get("output", [])
        if isinstance(output_items, list):
            chunks: list[str] = []
            for item in output_items:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                content_items = item.get("content", [])
                if not isinstance(content_items, list):
                    continue
                for content_item in content_items:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
            if chunks:
                return "\n".join(chunks)

        raise ValueError(f"Responses API 未返回可解析的文本内容: {data}")
