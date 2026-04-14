import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
from typing import Any, MutableMapping

import streamlit as st

from smartshop_rag.web.bootstrap import sync_session_state
from smartshop_rag.services.chat_service import ChatServiceError, run_chat


SessionState = MutableMapping[str, Any]
PROCESS_PANEL_CSS = """
<style>
.process-note-wrapper {
  color: #9ca3af;
  font-size: 0.9rem;
  line-height: 1.85;
}
.process-note-loading {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin-bottom: 0.45rem;
}
.process-note-spinner {
  width: 0.82rem;
  height: 0.82rem;
  border: 2px solid rgba(156, 163, 175, 0.3);
  border-top-color: #9ca3af;
  border-radius: 50%;
  animation: process-note-spin 0.9s linear infinite;
  flex-shrink: 0;
}
.process-note-item {
  margin: 0.18rem 0;
}
@keyframes process-note-spin {
  to { transform: rotate(360deg); }
}
</style>
"""


def render_chat_history(session_state: SessionState) -> None:
    for message in session_state["message"]:
        st.chat_message(message["role"]).write(message["content"])


def handle_chat_interaction(session_state: SessionState) -> None:
    prompt = st.chat_input("请输入电商客服问题")
    if not prompt:
        return

    st.chat_message("user").write(prompt)
    current_process_events: list[dict[str, str]] = []
    event_queue: Queue[dict[str, str]] = Queue()

    try:
        with st.chat_message("assistant"):
            with st.expander("查看处理过程", expanded=False):
                process_placeholder = st.empty()

            def render_process_notes(is_processing: bool = True) -> None:
                notes = build_process_notes(current_process_events)
                process_placeholder.markdown(
                    build_process_panel_html(notes, is_processing=is_processing),
                    unsafe_allow_html=True,
                )

            def on_status_event(event: dict[str, str]) -> None:
                event_queue.put(event)

            render_process_notes()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    run_chat,
                    user_id=session_state["selected_user_id"],
                    message=prompt,
                    session_id=session_state.get("current_session_id"),
                    status_event_callback=on_status_event,
                )

                while not future.done():
                    _drain_process_event_queue(event_queue, current_process_events)
                    render_process_notes()
                    time.sleep(0.05)

                _drain_process_event_queue(event_queue, current_process_events)
                render_process_notes(is_processing=False)
                result = future.result()
            st.write_stream(_yield_response_chunks(result["answer"]))
    except ChatServiceError as exc:
        st.error(str(exc))
        return

    session_state["latest_status_events"] = result["status_events"]
    sync_session_state(session_state, session_state["selected_user_id"], result["session_data"])
    st.rerun()


def build_process_notes(events: list[dict[str, str]]) -> list[str]:
    notes: list[str] = []
    for event in events:
        note = _build_process_note(event)
        if note and note not in notes:
            notes.append(note)
    return notes


def build_process_panel_html(notes: list[str], is_processing: bool = True) -> str:
    loading_line = (
        "<div class='process-note-loading'>"
        "<span class='process-note-spinner'></span>"
        "<span>正在检索知识并组织回复，请稍等片刻...</span>"
        "</div>"
        if is_processing
        else "<div class='process-note-item'>本轮处理过程已完成。</div>"
    )

    note_lines = notes or ["正在准备处理你的问题..."]
    notes_html = "".join(
        f"<div class='process-note-item'>{index}. {note}</div>"
        for index, note in enumerate(note_lines, start=1)
    )
    return f"{PROCESS_PANEL_CSS}<div class='process-note-wrapper'>{loading_line}{notes_html}</div>"


def _yield_response_chunks(text: str, chunk_size: int = 4, delay: float = 0.02):
    buffer = ""
    for char in text:
        buffer += char
        if char == "\n" or len(buffer) >= chunk_size:
            yield buffer
            time.sleep(delay)
            buffer = ""

    if buffer:
        yield buffer


def _drain_process_event_queue(
    event_queue: Queue[dict[str, str]],
    current_process_events: list[dict[str, str]],
) -> None:
    while True:
        try:
            event = event_queue.get_nowait()
        except Empty:
            return
        current_process_events.append(event)


def _build_process_note(event: dict[str, str]) -> str:
    event_type = (event.get("event_type") or "").strip()
    if event_type == "stage.rag":
        return "我在召回和当前商品、售后或规则相关的知识资料。"
    if event_type == "stage.final":
        return "我在根据检索到的资料整理最终回复。"
    if event_type == "error.agent":
        return "当前处理过程中出现异常，本次回复可能无法完整生成。"
    return ""
