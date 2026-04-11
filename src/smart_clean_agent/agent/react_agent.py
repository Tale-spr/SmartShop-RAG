from typing import Any

from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from smart_clean_agent.agent.runtime_context import AgentRuntimeContext
from smart_clean_agent.agent.tools.middleware import build_runtime_context_prompt
from smart_clean_agent.services.status_event_service import record_status_event
from smart_clean_agent.utils.prompt_loader import load_system_prompts


class ReactAgent:
    def __init__(self, model: BaseChatModel, tools: list[BaseTool]):
        self.chat_model = model
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def execute_stream(self, query: str, runtime_context: AgentRuntimeContext):
        yield self.execute(query, runtime_context)

    def execute(self, query: str, runtime_context: AgentRuntimeContext) -> str:
        rag_tool = self.tool_map.get("rag_summarize")
        if rag_tool is None:
            raise ValueError("未配置 rag_summarize 工具")

        runtime_context.setdefault("trace_tool_calls", [])
        runtime_context.setdefault("retrieval_trace", [])
        runtime_context.setdefault("retrieved_docs", [])

        record_status_event(
            runtime_context,
            event_type="stage.rag",
            title="正在检索知识库",
            detail="正在召回电商客服相关知识并整理参考证据",
        )
        runtime_context["trace_tool_calls"].append("rag_summarize")

        rag_summary = str(rag_tool.invoke({"query": query})).strip()
        docs = self._get_retrieved_docs(rag_tool)
        runtime_context["retrieved_docs"] = [self._serialize_doc(doc) for doc in docs]
        runtime_context["retrieval_trace"].append({"query": query, "doc_count": str(len(docs))})

        record_status_event(
            runtime_context,
            event_type="stage.final",
            title="正在生成最终回答",
            detail="正在结合检索结果与会话上下文组织最终回复",
        )
        return self._generate_answer(query=query, rag_summary=rag_summary, docs=docs, runtime_context=runtime_context).strip()

    def _generate_answer(
        self,
        *,
        query: str,
        rag_summary: str,
        docs: list[Document],
        runtime_context: AgentRuntimeContext,
    ) -> str:
        system_prompt = load_system_prompts()
        context_prompt = build_runtime_context_prompt(runtime_context)
        evidence_prompt = self._build_evidence_prompt(docs)
        user_prompt = (
            f"用户问题:\n{query}\n\n"
            f"{context_prompt}\n\n"
            f"知识库摘要:\n{rag_summary or '未检索到可靠摘要。'}\n\n"
            f"检索证据:\n{evidence_prompt}\n\n"
            "请基于检索到的资料直接回答用户问题。"
            "回答应自然、简洁、准确；如果资料不足以支撑明确结论，请明确说明不知道或资料不足。"
            "不要编造商品信息、政策条款、价格、库存、物流时效或售后规则。"
        ).strip()
        response = self.chat_model.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return getattr(response, "content", "") or ""

    def _build_evidence_prompt(self, docs: list[Document]) -> str:
        if not docs:
            return "未检索到有效参考资料。"

        lines: list[str] = []
        for index, doc in enumerate(docs[:4], start=1):
            source = str(doc.metadata.get("source", "")).strip()
            content = " ".join((doc.page_content or "").split())
            if len(content) > 220:
                content = content[:220].rstrip() + "..."
            prefix = f"资料{index}"
            if source:
                prefix += f" | 来源: {source}"
            lines.append(f"{prefix}\n{content}")
        return "\n\n".join(lines)

    def _get_retrieved_docs(self, rag_tool: BaseTool) -> list[Document]:
        rag_service = getattr(rag_tool, "_rag_service", None)
        docs = getattr(rag_service, "last_retrieved_docs", [])
        return docs if isinstance(docs, list) else []

    def _serialize_doc(self, doc: Document) -> dict[str, Any]:
        content = " ".join((doc.page_content or "").split())
        if len(content) > 220:
            content = content[:220].rstrip() + "..."
        return {
            "content": content,
            "metadata": {str(key): str(value) for key, value in (doc.metadata or {}).items()},
        }
