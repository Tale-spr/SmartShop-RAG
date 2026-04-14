def build_runtime_context_prompt(context: dict) -> str:
    session_summary = (context.get("session_summary") or "").strip()
    recent_history = (context.get("recent_history") or "").strip()
    if not session_summary and not recent_history:
        return ""

    sections = ["当前会话上下文"]
    if session_summary:
        sections.append(f"会话摘要:\n{session_summary}")
    if recent_history:
        sections.append(f"最近几轮对话:\n{recent_history}")
    sections.append("如果当前问题与历史上下文冲突，以当前用户最新问题和本轮检索证据为准。")
    return "\n\n".join(sections)
