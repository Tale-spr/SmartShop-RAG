"""RAG 总结服务：混合检索并基于重排后的结果生成知识摘要。"""
import json
import re
from typing import Any, Callable

from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from smartshop_rag.model.factory import create_chat_model
from smartshop_rag.rag.bm25_retriever import BM25Index
from smartshop_rag.rag.vector_store import VectorStoreService
from smartshop_rag.utils.config_handler import rag_conf
from smartshop_rag.utils.prompt_loader import load_query_rewrite_prompt, load_rag_prompts, load_rerank_prompt


class RagSummarizeService:
    def __init__(
        self,
        model: BaseChatModel | None = None,
        vector_store_service: VectorStoreService | None = None,
        rewrite_model: BaseChatModel | None = None,
        rerank_model: BaseChatModel | None = None,
    ):
        self.vector_store = vector_store_service or VectorStoreService()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = model or create_chat_model(role="rag_chat")
        self.rewrite_model = rewrite_model or create_chat_model(role="rewrite_chat")
        self.rerank_model = rerank_model or create_chat_model(role="rerank_chat")
        self.rewrite_prompt = PromptTemplate.from_template(load_query_rewrite_prompt())
        self.rerank_prompt = PromptTemplate.from_template(load_rerank_prompt())
        self.last_retrieved_docs: list[Document] = []
        self.last_retrieval_trace: dict[str, Any] = {}
        self.trace_callback: Callable[[str, list[Document]], None] | None = None
        self.chain = self.prompt_template | self.model | StrOutputParser()
        self.retrieval_conf = rag_conf.get("retrieval", {})
        self.default_mode = str(self.retrieval_conf.get("default_mode", "hybrid_rerank"))
        self.vector_top_k = int(self.retrieval_conf.get("vector_top_k", 6))
        self.bm25_top_k = int(self.retrieval_conf.get("bm25_top_k", 6))
        self.rerank_top_n = int(self.retrieval_conf.get("rerank_top_n", 4))
        self.rerank_candidate_limit = int(self.retrieval_conf.get("rerank_candidate_limit", 8))
        self._bm25_index: BM25Index | None = None
        self._all_chunked_documents: list[Document] | None = None
        self._model_pattern = re.compile(r"\bMF-[A-Z0-9]+\b", re.IGNORECASE)

    def _get_chunked_documents(self) -> list[Document]:
        if self._all_chunked_documents is None:
            self._all_chunked_documents = self.vector_store.load_all_chunked_documents()
        return self._all_chunked_documents

    def _get_bm25_index(self) -> BM25Index:
        if self._bm25_index is None:
            self._bm25_index = BM25Index(self._get_chunked_documents())
        return self._bm25_index

    def rewrite_query(self, query: str) -> str:
        prompt = self.rewrite_prompt.format(input=query)
        response = self.rewrite_model.invoke([HumanMessage(content=prompt)])
        rewritten = str(getattr(response, "content", "") or "").strip()
        return rewritten or query

    def _vector_retrieve(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        return self.vector_store.vector_search(query, top_k=top_k)

    def _bm25_retrieve(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        matches = self._get_bm25_index().search(query, top_k=top_k)
        return [
            {
                "document": match.document,
                "score": round(match.score, 6),
                "rank": match.rank,
                "source": "bm25",
            }
            for match in matches
        ]

    def _merge_results(self, vector_results: list[dict[str, Any]], bm25_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for result in vector_results + bm25_results:
            document: Document = result["document"]
            chunk_id = str(document.metadata.get("chunk_id") or document.metadata.get("source") or id(document))
            existing = merged.get(chunk_id)
            if existing is None:
                merged[chunk_id] = {
                    "document": document,
                    "score": result.get("score"),
                    "vector_rank": result.get("rank") if result["source"] == "vector" else None,
                    "bm25_rank": result.get("rank") if result["source"] == "bm25" else None,
                    "source": result["source"],
                }
                continue
            if result["source"] == "vector":
                existing["vector_rank"] = result.get("rank")
            if result["source"] == "bm25":
                existing["bm25_rank"] = result.get("rank")
                if existing.get("score") is None:
                    existing["score"] = result.get("score")
            if existing["source"] != result["source"]:
                existing["source"] = "both"
        merged_list = list(merged.values())
        merged_list.sort(key=lambda item: (item.get("vector_rank") or 9999, item.get("bm25_rank") or 9999))
        return merged_list[: self.vector_top_k + self.bm25_top_k]

    def _rerank_results(
        self,
        *,
        query: str,
        normalized_query: str,
        merged_results: list[dict[str, Any]],
        enabled: bool,
    ) -> list[dict[str, Any]]:
        candidates = merged_results[: self.rerank_candidate_limit]
        if not enabled or not candidates:
            return candidates[: self.rerank_top_n]

        candidate_lines: list[str] = []
        for index, item in enumerate(candidates, start=1):
            doc: Document = item["document"]
            content = " ".join((doc.page_content or "").split())
            if len(content) > 260:
                content = content[:260].rstrip() + "..."
            candidate_lines.append(
                f"候选ID: C{index}\n"
                f"来源类型: {item['source']}\n"
                f"文档类型: {doc.metadata.get('doc_type', '')}\n"
                f"型号: {doc.metadata.get('model', '')}\n"
                f"内容: {content}"
            )
        prompt = self.rerank_prompt.format(
            input=query,
            normalized_query=normalized_query,
            candidates="\n\n".join(candidate_lines),
        )
        response = self.rerank_model.invoke([HumanMessage(content=prompt)])
        ranked_ids = self._parse_ranked_ids(str(getattr(response, "content", "") or ""))
        if not ranked_ids:
            return candidates[: self.rerank_top_n]
        mapped = {f"C{index}": item for index, item in enumerate(candidates, start=1)}
        ranked: list[dict[str, Any]] = []
        for candidate_id in ranked_ids:
            item = mapped.get(candidate_id)
            if item is not None:
                ranked.append(item)
        if not ranked:
            return candidates[: self.rerank_top_n]
        for item in candidates:
            if item not in ranked:
                ranked.append(item)
        return ranked[: self.rerank_top_n]

    def _parse_ranked_ids(self, content: str) -> list[str]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return []
        ranked_ids = payload.get("ranked_ids")
        if not isinstance(ranked_ids, list):
            return []
        return [str(item) for item in ranked_ids if str(item).strip()]

    def _extract_query_models(self, *texts: str) -> list[str]:
        seen: set[str] = set()
        detected: list[str] = []
        for text in texts:
            if not text:
                continue
            for match in self._model_pattern.findall(text):
                normalized = match.upper()
                if normalized in seen:
                    continue
                seen.add(normalized)
                detected.append(normalized)
        return detected

    def _extract_retrieved_models(self, final_results: list[dict[str, Any]]) -> list[str]:
        seen: set[str] = set()
        models: list[str] = []
        for item in final_results:
            doc: Document = item["document"]
            model = str(doc.metadata.get("model", "")).strip().upper()
            if not model or model == "SHARED":
                continue
            if model in seen:
                continue
            seen.add(model)
            models.append(model)
        return models

    def _determine_model_confirmation_status(
        self,
        *,
        detected_query_models: list[str],
        retrieved_models: list[str],
    ) -> str:
        if not detected_query_models:
            return "unconfirmed"
        detected_set = set(detected_query_models)
        retrieved_set = set(retrieved_models)
        if not retrieved_set:
            return "ambiguous"
        if retrieved_set.issubset(detected_set):
            return "confirmed"
        return "ambiguous"

    def _determine_model_confirmation_source(
        self,
        *,
        detected_query_models: list[str],
        retrieved_models: list[str],
        model_confirmation_status: str,
    ) -> str:
        if not detected_query_models:
            return "retrieval_inferred"
        if model_confirmation_status == "confirmed":
            return "explicit_query"
        return "conflicted"

    def _should_reconfirm_model(
        self,
        *,
        model_confirmation_source: str,
    ) -> bool:
        return model_confirmation_source in {"retrieval_inferred", "conflicted"}

    def retrieve_docs(self, query: str, *, mode: str | None = None) -> list[Document]:
        actual_mode = mode or self.default_mode
        normalized_query = self.rewrite_query(query)
        detected_query_models = self._extract_query_models(query, normalized_query)
        vector_results = self._vector_retrieve(normalized_query, top_k=self.vector_top_k) if actual_mode in {"vector", "hybrid", "hybrid_rerank"} else []
        bm25_results = self._bm25_retrieve(normalized_query, top_k=self.bm25_top_k) if actual_mode in {"bm25", "hybrid", "hybrid_rerank"} else []
        if actual_mode == "vector":
            merged_results = vector_results[: self.rerank_top_n]
            final_results = merged_results
        elif actual_mode == "bm25":
            merged_results = bm25_results[: self.rerank_top_n]
            final_results = merged_results
        else:
            merged_results = self._merge_results(vector_results, bm25_results)
            final_results = self._rerank_results(
                query=query,
                normalized_query=normalized_query,
                merged_results=merged_results,
                enabled=(actual_mode == "hybrid_rerank"),
            )
        docs = [item["document"] for item in final_results]
        retrieved_models = self._extract_retrieved_models(final_results)
        model_confirmation_status = self._determine_model_confirmation_status(
            detected_query_models=detected_query_models,
            retrieved_models=retrieved_models,
        )
        model_confirmation_source = self._determine_model_confirmation_source(
            detected_query_models=detected_query_models,
            retrieved_models=retrieved_models,
            model_confirmation_status=model_confirmation_status,
        )
        should_reconfirm_model = self._should_reconfirm_model(
            model_confirmation_source=model_confirmation_source,
        )
        confirmed_model = detected_query_models[0] if model_confirmation_source == "explicit_query" and detected_query_models else ""
        self.last_retrieved_docs = docs
        self.last_retrieval_trace = {
            "query": query,
            "normalized_query": normalized_query,
            "mode": actual_mode,
            "detected_query_models": detected_query_models,
            "retrieved_models": retrieved_models,
            "model_confirmation_status": model_confirmation_status,
            "model_confirmation_source": model_confirmation_source,
            "should_reconfirm_model": should_reconfirm_model,
            "confirmed_model": confirmed_model,
            "vector_hit_count": len(vector_results),
            "bm25_hit_count": len(bm25_results),
            "merged_candidate_count": len(merged_results),
            "rerank_selected_count": len(final_results),
            "doc_count": str(len(docs)),
            "final_docs": [
                {
                    "chunk_id": str(item["document"].metadata.get("chunk_id", "")),
                    "source": item["source"],
                    "rank": index,
                    "doc_type": str(item["document"].metadata.get("doc_type", "")),
                    "model": str(item["document"].metadata.get("model", "")),
                    "score": item.get("score"),
                }
                for index, item in enumerate(final_results, start=1)
            ],
        }
        if self.trace_callback is not None:
            self.trace_callback(query, docs)
        return docs

    def rag_summarize(self, query: str, *, mode: str | None = None) -> str:
        context_docs = self.retrieve_docs(query, mode=mode)
        context_parts = []
        for index, doc in enumerate(context_docs, start=1):
            context_parts.append(f"【参考资料{index}】{doc.page_content} | 元数据: {doc.metadata}")
        return self.chain.invoke({"input": query, "context": "\n".join(context_parts)})
