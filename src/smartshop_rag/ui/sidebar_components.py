from typing import Any, MutableMapping

import streamlit as st

from smartshop_rag.web.bootstrap import sync_session_state
from smartshop_rag.services.session_service import create_session, delete_session, list_sessions, load_session


SessionState = MutableMapping[str, Any]


def render_user_profile_sidebar(session_state: SessionState) -> None:
    st.subheader("当前访客")
    current_user_id = session_state.get("selected_user_id", "demo_user")
    input_user_id = st.text_input("用户ID", value=current_user_id, help="用于区分不同会话空间")
    normalized_user_id = (input_user_id or "").strip() or "demo_user"
    if normalized_user_id != current_user_id:
        session_state["selected_user_id"] = normalized_user_id
        st.rerun()


def render_session_sidebar(session_state: SessionState) -> None:
    current_user_id = session_state["selected_user_id"]

    st.subheader("会话管理")
    if st.button("新建会话", use_container_width=True):
        session_data = create_session(current_user_id)
        sync_session_state(session_state, current_user_id, session_data)
        st.rerun()

    session_list = list_sessions(current_user_id)
    for session in session_list:
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                session["title"],
                key=f"load_{current_user_id}_{session['session_id']}",
                use_container_width=True,
                type="primary" if session["session_id"] == session_state.get("current_session_id") else "secondary",
            ):
                loaded_session = load_session(current_user_id, session["session_id"])
                if loaded_session is not None:
                    sync_session_state(session_state, current_user_id, loaded_session)
                    st.rerun()

        with col2:
            if st.button("删", key=f"delete_{current_user_id}_{session['session_id']}", use_container_width=True):
                delete_session(current_user_id, session["session_id"])
                remaining_sessions = list_sessions(current_user_id)
                if session_state.get("current_session_id") == session["session_id"]:
                    next_session = remaining_sessions[0] if remaining_sessions else create_session(current_user_id)
                    sync_session_state(session_state, current_user_id, next_session)
                st.rerun()

    st.divider()
    st.caption("当前会话摘要")
    st.caption(session_state.get("current_session_summary") or "暂无会话摘要")
