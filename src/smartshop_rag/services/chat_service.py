from collections.abc import Callable
from functools import lru_cache
from typing import Any

from smartshop_rag.agent.react_agent import ReactAgent
from smartshop_rag.agent.runtime_context import AgentRuntimeContext
from smartshop_rag.services.conversation_memory_service import build_recent_history, summarize_messages
from smartshop_rag.services.session_service import create_session, load_session, save_session
from smartshop_rag.services.status_event_service import get_visible_status_events
from smartshop_rag.web.bootstrap import build_agent
from smartshop_rag.utils.logger_handler import logger


class ChatServiceError(Exception):
    def __init__(self, message: str, code: str = "chat_service_error", status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


@lru_cache(maxsize=1)
def get_chat_agent() -> ReactAgent:
    return build_agent()


def run_chat(
    user_id: str,
    message: str,
    session_id: str | None = None,
    agent: ReactAgent | None = None,
    status_event_callback: Callable[[dict[str, str]], Any] | None = None,
) -> dict[str, Any]:
    normalized_user_id = (user_id or "").strip()
    normalized_message = (message or "").strip()
    if not normalized_user_id:
        raise ChatServiceError("user_id 不能为空", code="invalid_user_id", status_code=422)
    if not normalized_message:
        raise ChatServiceError("message 不能为空", code="invalid_message", status_code=422)

    session_data = _load_target_session(normalized_user_id, session_id)
    session_data["messages"] = list(session_data.get("messages", [])) + [{"role": "user", "content": normalized_message}]
    status_events: list[dict[str, str]] = []
    runtime_context = _build_runtime_context(
        user_id=normalized_user_id,
        session_data=session_data,
        status_events=status_events,
        status_event_callback=status_event_callback,
    )

    runtime_agent = agent or get_chat_agent()
    try:
        answer = runtime_agent.execute(normalized_message, runtime_context)
    except Exception as exc:
        logger.error(f"[ChatService]请求执行失败: {str(exc)}", exc_info=True)
        raise ChatServiceError("RAG 问答执行失败", code="agent_execution_failed", status_code=500) from exc

    session_data["messages"].append({"role": "assistant", "content": answer})
    saved_session = save_session(normalized_user_id, session_data)

    return {
        "user_id": normalized_user_id,
        "session_id": saved_session["session_id"],
        "answer": answer,
        "status_events": get_visible_status_events(status_events),
        "session_summary": saved_session.get("session_summary", ""),
        "session_data": saved_session,
        "retrieval_trace": runtime_context.get("retrieval_trace", []),
        "retrieved_docs": runtime_context.get("retrieved_docs", []),
    }


def _load_target_session(user_id: str, session_id: str | None) -> dict[str, Any]:
    normalized_session_id = (session_id or "").strip()
    if normalized_session_id:
        session_data = load_session(user_id, normalized_session_id)
        if session_data is None:
            raise ChatServiceError("指定会话不存在", code="session_not_found", status_code=404)
        return session_data
    return create_session(user_id)


def _build_runtime_context(
    *,
    user_id: str,
    session_data: dict[str, Any],
    status_events: list[dict[str, str]],
    status_event_callback: Callable[[dict[str, str]], Any] | None,
) -> AgentRuntimeContext:
    messages = session_data.get("messages", [])
    runtime_context: AgentRuntimeContext = {
        "user_id": user_id,
        "session_id": session_data["session_id"],
        "session_summary": summarize_messages(messages),
        "recent_history": build_recent_history(messages),
        "trace_tool_calls": [],
        "retrieval_trace": [],
        "retrieved_docs": [],
        "status_events": status_events,
    }
    if status_event_callback is not None:
        runtime_context["status_event_callback"] = status_event_callback
    return runtime_context
