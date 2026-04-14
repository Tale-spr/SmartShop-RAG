from collections.abc import Callable
from typing import Any, NotRequired, TypedDict


class AgentRuntimeContext(TypedDict):
    user_id: str
    session_id: str
    session_summary: str
    recent_history: str
    trace_tool_calls: NotRequired[list[str]]
    retrieval_trace: NotRequired[list[dict[str, Any]]]
    retrieved_docs: NotRequired[list[dict[str, Any]]]
    status_events: NotRequired[list[dict[str, str]]]
    status_event_callback: NotRequired[Callable[[dict[str, str]], Any]]
