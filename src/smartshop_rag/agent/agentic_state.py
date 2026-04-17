from typing import Any, Literal, TypedDict


class AgenticState(TypedDict, total=False):
    question: str
    user_id: str
    session_id: str

    session_summary: str
    recent_history: str

    intent: Literal[
        "smalltalk",
        "capability_query",
        "non_domain",
        "product_qa",
        "policy_qa",
        "usage_qa",
        "fault_qa",
    ]

    rewritten_query: str
    transformed_query: str
    active_query: str

    documents: list[Any]
    retrieved_docs: list[dict[str, Any]]
    retrieval_trace: list[dict[str, Any]]

    model_confirmation_status: str
    model_confirmation_source: str
    should_reconfirm_model: bool
    confirmed_model: str
    detected_query_models: list[str]
    retrieved_models: list[str]

    retrieval_decision: Literal["enough", "retry", "fallback"]
    answer_decision: Literal["answer", "clarify", "fallback"]

    retry_count: int
    max_retry: int

    answer: str
    final_answer: str

    trace_tool_calls: list[str]
    status_events: list[dict[str, str]]
    status_event_callback: Any
