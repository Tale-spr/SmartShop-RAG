from langchain_core.tools import BaseTool, tool

from smart_clean_agent.rag.rag_service import RagSummarizeService


def build_rag_summarize_tool(rag_service: RagSummarizeService) -> BaseTool:
    @tool(description="从本地知识库中检索与当前电商客服问题相关的参考资料并生成摘要")
    def rag_summarize(query: str) -> str:
        return rag_service.rag_summarize(query)

    setattr(rag_summarize, "_rag_service", rag_service)
    return rag_summarize


def create_agent_tools(rag_service: RagSummarizeService) -> list[BaseTool]:
    return [build_rag_summarize_tool(rag_service)]
