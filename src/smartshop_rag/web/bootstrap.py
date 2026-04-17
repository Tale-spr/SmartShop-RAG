from dataclasses import dataclass
from typing import Any, MutableMapping

from smartshop_rag.agent.react_agent import ReactAgent
from smartshop_rag.agent.runtime_context import AgentRuntimeContext
from smartshop_rag.agent.tools.agent_tools import create_agent_tools
from smartshop_rag.model.factory import create_chat_model, create_embedding_model
from smartshop_rag.rag.rag_service import RagSummarizeService
from smartshop_rag.rag.vector_store import VectorStoreService
from smartshop_rag.services.dependency_service import validate_runtime_dependencies
from smartshop_rag.services.session_service import create_session, get_latest_session, load_session, save_session


@dataclass
class AppContext:
    selected_user_id: str


def validate_online_dependencies() -> None:
    validate_runtime_dependencies()


def build_agent() -> ReactAgent:
    chat_model = create_chat_model()
    try:
        smalltalk_model = create_chat_model(role='smalltalk_chat')
    except ValueError:
        smalltalk_model = None
    embedding_model = create_embedding_model()
    vector_store_service = VectorStoreService(embedding_function=embedding_model)
    rag_service = RagSummarizeService(
        model=create_chat_model(role='rag_chat'),
        rewrite_model=create_chat_model(role='rewrite_chat'),
        rerank_model=create_chat_model(role='rerank_chat'),
        vector_store_service=vector_store_service,
    )
    tools = create_agent_tools(rag_service)
    return ReactAgent(model=chat_model, tools=tools, rag_service=rag_service, smalltalk_model=smalltalk_model)


def get_or_create_agent(session_state: MutableMapping[str, Any]) -> ReactAgent:
    agent = session_state.get('agent')
    if agent is None:
        agent = build_agent()
        session_state['agent'] = agent
    return agent


def initialize_ui_state(session_state: MutableMapping[str, Any]) -> None:
    session_state.setdefault('message', [])
    session_state.setdefault('current_status_events', [])
    session_state.setdefault('latest_status_events', [])
    session_state.setdefault('selected_user_id', 'demo_user')


def sync_session_state(session_state: MutableMapping[str, Any], user_id: str, session_data: dict) -> None:
    session_state['current_user_id'] = user_id
    session_state['current_session_id'] = session_data['session_id']
    session_state['current_session_title'] = session_data['title']
    session_state['current_session_created_at'] = session_data['created_at']
    session_state['current_session_summary'] = session_data.get('session_summary', '')
    session_state['current_recent_history'] = session_data.get('recent_history', '')
    session_state['message'] = session_data['messages']


def ensure_active_session(session_state: MutableMapping[str, Any], user_id: str) -> None:
    latest_session = get_latest_session(user_id)
    if latest_session is None:
        latest_session = create_session(user_id)
    sync_session_state(session_state, user_id, latest_session)


def save_current_session(session_state: MutableMapping[str, Any]) -> None:
    current_session_id = session_state.get('current_session_id')
    current_user_id = session_state.get('current_user_id')
    if not current_session_id or not current_user_id:
        return
    session_data = {
        'session_id': current_session_id,
        'title': session_state.get('current_session_title', current_session_id),
        'created_at': session_state.get('current_session_created_at'),
        'messages': session_state.get('message', []),
    }
    saved_session = save_session(current_user_id, session_data)
    session_state['current_session_title'] = saved_session['title']
    session_state['current_session_created_at'] = saved_session['created_at']
    session_state['current_session_summary'] = saved_session.get('session_summary', '')
    session_state['current_recent_history'] = saved_session.get('recent_history', '')


def build_runtime_context(session_state: MutableMapping[str, Any], status_events: list[dict[str, str]] | None = None) -> AgentRuntimeContext:
    return {
        'user_id': session_state['selected_user_id'],
        'session_id': session_state.get('current_session_id', ''),
        'session_summary': session_state.get('current_session_summary', ''),
        'recent_history': session_state.get('current_recent_history', ''),
        'trace_tool_calls': [],
        'retrieval_trace': [],
        'retrieved_docs': [],
        'status_events': status_events or [],
    }


def build_app_context(session_state: MutableMapping[str, Any]) -> AppContext:
    initialize_ui_state(session_state)
    selected_user_id = (session_state.get('selected_user_id') or 'demo_user').strip() or 'demo_user'
    session_state['selected_user_id'] = selected_user_id
    if session_state.get('current_user_id') != selected_user_id:
        ensure_active_session(session_state, selected_user_id)
    elif not session_state.get('current_session_id'):
        ensure_active_session(session_state, selected_user_id)
    elif load_session(selected_user_id, session_state['current_session_id']) is None:
        ensure_active_session(session_state, selected_user_id)
    return AppContext(selected_user_id=selected_user_id)
