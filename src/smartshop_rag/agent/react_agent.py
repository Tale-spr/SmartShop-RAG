from typing import Any

from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from smartshop_rag.agent.runtime_context import AgentRuntimeContext
from smartshop_rag.agent.tools.middleware import build_runtime_context_prompt
from smartshop_rag.services.status_event_service import record_status_event
from smartshop_rag.utils.prompt_loader import load_system_prompts


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
            detail="正在进行 query rewrite、混合召回和重排",
        )
        runtime_context["trace_tool_calls"].append("rag_summarize")

        rag_summary = str(rag_tool.invoke({"query": query})).strip()
        docs = self._get_retrieved_docs(rag_tool)
        runtime_context["retrieved_docs"] = [self._serialize_doc(doc) for doc in docs]
        trace = self._get_retrieval_trace(rag_tool, query, docs)
        runtime_context["retrieval_trace"].append(trace)

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
        model_guard_prompt = self._build_model_confirmation_prompt(runtime_context)
        user_prompt = (
            f"用户问题:\n{query}\n\n"
            f"{context_prompt}\n\n"
            f"知识库摘要:\n{rag_summary or '未检索到可靠摘要。'}\n\n"
            f"检索证据:\n{evidence_prompt}\n\n"
            f"型号确认约束:\n{model_guard_prompt}\n\n"
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

    def _build_model_confirmation_prompt(self, runtime_context: AgentRuntimeContext) -> str:
        trace = self._get_latest_retrieval_trace(runtime_context)
        if not trace:
            return "当前没有额外的型号确认信息；如资料不足，请保持保守表达。"

        status = str(trace.get("model_confirmation_status", "")).strip() or "unconfirmed"
        source = str(trace.get("model_confirmation_source", "")).strip() or "retrieval_inferred"
        should_reconfirm = bool(trace.get("should_reconfirm_model", True))
        confirmed_model = str(trace.get("confirmed_model", "")).strip()
        query_models = trace.get("detected_query_models") or []
        retrieved_models = trace.get("retrieved_models") or []
        query_models_text = "、".join(str(item) for item in query_models if str(item).strip()) or "未在问题中识别到明确型号"
        retrieved_models_text = "、".join(str(item) for item in retrieved_models if str(item).strip()) or "当前证据未形成稳定型号集合"

        if not should_reconfirm:
            confirmed_model_text = confirmed_model or query_models_text
            return (
                f"型号确认状态: {status}。确认来源: {source}。"
                f"已确认型号: {confirmed_model_text}。"
                f"当前证据涉及的型号: {retrieved_models_text}。"
                "用户已经明确提供型号，且当前证据与之不冲突。"
                "请直接基于该型号回答参数、容量、适用人数或功能差异。"
                "不要再次建议用户查看铭牌、说明书封面、订单页或再次确认型号。"
                "不要使用“建议再确认型号”“您可以再核对型号”这类重复核验话术。"
                "只有在当前回答确实涉及证据冲突时，才允许补一句“如页面信息有更新，以最新资料为准”。"
            )

        return (
            f"型号确认状态: {status}。确认来源: {source}。"
            f"问题中明确出现的型号: {query_models_text}。"
            f"当前证据涉及的候选型号: {retrieved_models_text}。"
            "当前不能把任何检索到的型号直接当作用户真实购买的型号。"
            "禁止使用“您这款就是MF-XXX”或“您买的就是MF-XXX”这类确定表述。"
            "如需引用候选型号，只能用“候选型号可能包括……”或“如果您说的是MF-XXX”这类表达。"
            "优先提示用户查看机身铭牌、说明书封面、订单页，或提供颜色、旋钮/面板样式、是否带可视窗等外观特征来辅助确认。"
        )

    def _get_latest_retrieval_trace(self, runtime_context: AgentRuntimeContext) -> dict[str, Any] | None:
        traces = runtime_context.get("retrieval_trace") or []
        if not isinstance(traces, list) or not traces:
            return None
        latest = traces[-1]
        return latest if isinstance(latest, dict) else None

    def _get_retrieved_docs(self, rag_tool: BaseTool) -> list[Document]:
        rag_service = getattr(rag_tool, "_rag_service", None)
        docs = getattr(rag_service, "last_retrieved_docs", [])
        return docs if isinstance(docs, list) else []

    def _get_retrieval_trace(self, rag_tool: BaseTool, query: str, docs: list[Document]) -> dict[str, Any]:
        rag_service = getattr(rag_tool, "_rag_service", None)
        trace = getattr(rag_service, "last_retrieval_trace", None)
        if isinstance(trace, dict) and trace:
            return trace
        return {"query": query, "doc_count": str(len(docs))}

    def _serialize_doc(self, doc: Document) -> dict[str, Any]:
        content = " ".join((doc.page_content or "").split())
        if len(content) > 220:
            content = content[:220].rstrip() + "..."
        return {
            "content": content,
            "metadata": {str(key): str(value) for key, value in (doc.metadata or {}).items()},
        }
