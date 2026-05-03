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
from smartshop_rag.utils.prompt_loader import (
    load_query_rewrite_prompt,
    load_rag_prompts,
    load_rerank_prompt,
    load_transform_query_prompt,
)


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
        self.transform_query_prompt = PromptTemplate.from_template(load_transform_query_prompt())
        self.last_retrieved_docs: list[Document] = []
        self.last_retrieval_trace: dict[str, Any] = {}
        self.trace_callback: Callable[[str, list[Document]], None] | None = None
        self.chain = self.prompt_template | self.model | StrOutputParser()
        self.retrieval_conf = rag_conf.get("retrieval", {})
        self.default_mode = str(self.retrieval_conf.get("default_mode", "hybrid_rerank"))
        self.vector_top_k = int(self.retrieval_conf.get("vector_top_k", 6))
        self.bm25_top_k = int(self.retrieval_conf.get("bm25_top_k", 6))
        self.vector_weight = float(self.retrieval_conf.get("vector_weight", 0.7))
        self.bm25_weight = float(self.retrieval_conf.get("bm25_weight", 0.3))
        self.rrf_k = int(self.retrieval_conf.get("rrf_k", 60))
        self.bm25_top_k_v2 = int(self.retrieval_conf.get("bm25_top_k_v2", 4))
        self.weighted_rrf_v2_conf = dict(self.retrieval_conf.get("weighted_rrf_v2", {}))
        self.model_mismatch_penalty = float(self.weighted_rrf_v2_conf.get("model_mismatch_penalty", 0.5))
        self.manual_bias_boost = float(self.weighted_rrf_v2_conf.get("manual_bias_boost", 1.1))
        self.weighted_rrf_v2_bucket_conf = {
            "explicit_model": self._load_v2_bucket_conf("explicit_model", 0.85, 0.15, 60),
            "weak_feature": self._load_v2_bucket_conf("weak_feature", 0.60, 0.40, 60),
            "generic": self._load_v2_bucket_conf("generic", 0.75, 0.25, 60),
        }
        self.rerank_top_n = int(self.retrieval_conf.get("rerank_top_n", 4))
        self.rerank_candidate_limit = int(self.retrieval_conf.get("rerank_candidate_limit", 8))
        self.context_postprocess_conf = dict(self.retrieval_conf.get("context_postprocess", {}))
        self.context_postprocess_enabled = bool(self.context_postprocess_conf.get("enabled", True))
        context_window_conf = dict(self.context_postprocess_conf.get("context_window", {}))
        self.context_window_target_chars = int(context_window_conf.get("target_chars", 650))
        self.context_window_neighbor_window = int(context_window_conf.get("neighbor_window", 1))
        self.context_window_max_expanded_docs = int(context_window_conf.get("max_expanded_docs", 4))
        section_role_conf = dict(self.context_postprocess_conf.get("section_role_weighting", {}))
        self.query_hint_penalty = float(section_role_conf.get("query_hint_penalty", 0.70))
        self.evidence_boost = float(section_role_conf.get("evidence_boost", 1.08))
        entity_coverage_conf = dict(self.context_postprocess_conf.get("entity_coverage", {}))
        self.entity_coverage_enabled = bool(entity_coverage_conf.get("enabled", True))
        self.entity_coverage_max_supplemental_docs = int(entity_coverage_conf.get("max_supplemental_docs", 3))
        self._bm25_index: BM25Index | None = None
        self._all_chunked_documents: list[Document] | None = None
        self._chunk_by_id: dict[str, Document] | None = None
        self._chunks_by_source: dict[str, list[Document]] | None = None
        self._chunks_by_model: dict[str, list[Document]] | None = None
        self._model_alias_map: dict[str, str] | None = None
        self._model_pattern = re.compile(r"\bMF-[A-Z0-9]+\b", re.IGNORECASE)
        self._weak_feature_pattern = re.compile(r"\b(?:\d+(?:\.\d+)?L|\d{4})\b", re.IGNORECASE)
        self._weak_feature_keywords = {
            "可视窗",
            "旋钮",
            "双热源",
            "自动断电",
            "触控",
            "按键",
            "电子可视",
            "方形烤篮",
            "圆形烤篮",
        }
        self._manual_intent_keywords = {
            "首次使用",
            "第一次用",
            "不工作",
            "推不进去",
            "异响",
            "白烟",
            "清洁",
            "怎么检查",
            "怎么处理",
            "怎么用",
            "怎么清洗",
            "故障",
            "排查",
        }
        self._entity_attribute_keywords = {
            "容量",
            "功率",
            "可视窗",
            "双热源",
            "双可视",
            "旋钮",
            "触控",
            "按键",
            "清洗",
            "故障",
            "白烟",
            "异响",
            "首次使用",
            "第一次",
            "不工作",
        }

    def _load_v2_bucket_conf(self, bucket_name: str, default_vector_weight: float, default_bm25_weight: float, default_rrf_k: int) -> dict[str, float | int]:
        bucket_conf = dict(self.weighted_rrf_v2_conf.get(bucket_name, {}))
        return {
            "vector_weight": float(bucket_conf.get("vector_weight", default_vector_weight)),
            "bm25_weight": float(bucket_conf.get("bm25_weight", default_bm25_weight)),
            "rrf_k": int(bucket_conf.get("rrf_k", default_rrf_k)),
        }

    def _get_chunked_documents(self) -> list[Document]:
        if self._all_chunked_documents is None:
            self._all_chunked_documents = self.vector_store.load_all_chunked_documents()
        return self._all_chunked_documents

    def _get_bm25_index(self) -> BM25Index:
        if self._bm25_index is None:
            self._bm25_index = BM25Index(self._get_chunked_documents())
        return self._bm25_index

    def _ensure_manifest_indexes(self) -> None:
        if self._chunk_by_id is not None:
            return
        chunk_by_id: dict[str, Document] = {}
        chunks_by_source: dict[str, list[Document]] = {}
        chunks_by_model: dict[str, list[Document]] = {}
        model_alias_map: dict[str, str] = {}
        for doc in self._get_chunked_documents():
            metadata = doc.metadata
            chunk_id = str(metadata.get("chunk_id") or "")
            if chunk_id:
                chunk_by_id[chunk_id] = doc
            source_path = str(metadata.get("source_path") or metadata.get("source") or "")
            if source_path:
                chunks_by_source.setdefault(source_path, []).append(doc)
            model = str(metadata.get("model") or "").strip().upper()
            if model and model != "SHARED":
                chunks_by_model.setdefault(model, []).append(doc)
                for alias in self._build_model_aliases(model):
                    model_alias_map.setdefault(alias, model)
        for docs in chunks_by_source.values():
            docs.sort(key=self._chunk_sort_key)
        for docs in chunks_by_model.values():
            docs.sort(key=self._chunk_sort_key)
        self._chunk_by_id = chunk_by_id
        self._chunks_by_source = chunks_by_source
        self._chunks_by_model = chunks_by_model
        self._model_alias_map = model_alias_map

    def _build_model_aliases(self, model: str) -> set[str]:
        normalized = model.upper()
        aliases = {normalized}
        suffix = normalized.split("-")[-1]
        aliases.add(suffix)
        aliases.add(re.sub(r"^KZ[CE]?", "", suffix))
        return {alias for alias in aliases if alias}

    def _chunk_sort_key(self, doc: Document) -> tuple[str, int]:
        source_path = str(doc.metadata.get("source_path") or doc.metadata.get("source") or "")
        try:
            chunk_index = int(doc.metadata.get("chunk_index", 0))
        except (TypeError, ValueError):
            chunk_index = 0
        return source_path, chunk_index

    def _get_chunk_id(self, doc: Document) -> str:
        return str(doc.metadata.get("chunk_id") or doc.metadata.get("source") or id(doc))

    def rewrite_query(self, query: str) -> str:
        prompt = self.rewrite_prompt.format(input=query)
        response = self.rewrite_model.invoke([HumanMessage(content=prompt)])
        rewritten = str(getattr(response, "content", "") or "").strip()
        return rewritten or query

    def transform_query(
        self,
        *,
        question: str,
        current_query: str,
        session_summary: str = "",
        recent_history: str = "",
        retrieval_summary: str = "",
    ) -> str:
        prompt = self.transform_query_prompt.format(
            question=question,
            current_query=current_query,
            session_summary=session_summary or "无",
            recent_history=recent_history or "无",
            retrieval_summary=retrieval_summary or "上一次检索未命中足够资料。",
        )
        response = self.rewrite_model.invoke([HumanMessage(content=prompt)])
        transformed = str(getattr(response, "content", "") or "").strip()
        return transformed or current_query

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
                    "rrf_score": None,
                    "base_rrf_score": None,
                    "adjusted_rrf_score": None,
                    "model_match": None,
                    "model_consistency_penalty_applied": False,
                    "manual_bias_applied": False,
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

    def _build_weighted_rrf_results(
        self,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        *,
        vector_weight: float,
        bm25_weight: float,
        rrf_k: int,
        candidate_limit: int,
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for result in vector_results + bm25_results:
            document: Document = result["document"]
            chunk_id = str(document.metadata.get("chunk_id") or document.metadata.get("source") or id(document))
            existing = fused.get(chunk_id)
            if existing is None:
                existing = {
                    "document": document,
                    "score": result.get("score"),
                    "vector_rank": None,
                    "bm25_rank": None,
                    "source": result["source"],
                    "base_rrf_score": 0.0,
                    "adjusted_rrf_score": 0.0,
                    "rrf_score": 0.0,
                    "model_match": None,
                    "model_consistency_penalty_applied": False,
                    "manual_bias_applied": False,
                }
                fused[chunk_id] = existing
            if result["source"] == "vector":
                rank = int(result.get("rank") or 0)
                existing["vector_rank"] = rank
                existing["base_rrf_score"] += vector_weight / (rrf_k + rank)
            if result["source"] == "bm25":
                rank = int(result.get("rank") or 0)
                existing["bm25_rank"] = rank
                existing["score"] = result.get("score") if existing.get("score") is None else existing.get("score")
                existing["base_rrf_score"] += bm25_weight / (rrf_k + rank)
            existing["adjusted_rrf_score"] = existing["base_rrf_score"]
            existing["rrf_score"] = existing["adjusted_rrf_score"]
            if existing["source"] != result["source"]:
                existing["source"] = "both"
        fused_list = list(fused.values())
        fused_list.sort(key=lambda item: item.get("adjusted_rrf_score", 0.0), reverse=True)
        return fused_list[:candidate_limit]

    def _weighted_rrf_results(self, vector_results: list[dict[str, Any]], bm25_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._build_weighted_rrf_results(
            vector_results,
            bm25_results,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
            rrf_k=self.rrf_k,
            candidate_limit=self.vector_top_k + self.bm25_top_k,
        )

    def _section_role(self, doc: Document) -> str:
        heading_path = str(doc.metadata.get("heading_path") or "")
        doc_type = str(doc.metadata.get("doc_type") or "").lower()
        if "典型问法映射" in heading_path:
            return "query_hint"
        if "核心定位" in heading_path:
            return "overview"
        evidence_terms = (
            "规格参数",
            "核心卖点",
            "适用场景",
            "快速入门",
            "清洁保养",
            "常见问题",
            "服务支持",
            "原始要点",
            "客服可直接引用",
            "发货",
            "退换货",
            "保修",
            "发票",
        )
        if any(term in heading_path for term in evidence_terms):
            return "evidence"
        if doc_type in {"specs", "service", "invoice", "returns", "shipping", "warranty"}:
            return "evidence"
        return "generic"

    def _apply_section_role_weighting(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not getattr(self, "context_postprocess_enabled", True):
            return results
        adjusted_results = list(results)
        for item in adjusted_results:
            doc: Document = item["document"]
            role = self._section_role(doc)
            weight = 1.0
            if role == "query_hint":
                weight = getattr(self, "query_hint_penalty", 0.70)
            elif role == "evidence":
                weight = getattr(self, "evidence_boost", 1.08)
            item["section_role"] = role
            item["section_role_weight"] = weight
            score = item.get("adjusted_rrf_score")
            if isinstance(score, (int, float)):
                item["adjusted_rrf_score"] = float(score) * weight
                item["rrf_score"] = item["adjusted_rrf_score"]
        if any(isinstance(item.get("adjusted_rrf_score"), (int, float)) for item in adjusted_results):
            adjusted_results.sort(key=lambda item: item.get("adjusted_rrf_score", 0.0), reverse=True)
        return adjusted_results

    def _extract_query_entities(self, *texts: str) -> dict[str, list[str]]:
        self._ensure_manifest_indexes()
        combined = "\n".join(text for text in texts if text)
        detected_models = self._extract_query_models(combined)
        model_set = {model.upper() for model in detected_models}
        alias_map = self._model_alias_map or {}
        upper_combined = combined.upper()
        for alias in sorted(alias_map, key=len, reverse=True):
            if len(alias) < 3:
                continue
            if re.search(rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])", upper_combined):
                model_set.add(alias_map[alias])
        attributes = sorted(keyword for keyword in getattr(self, "_entity_attribute_keywords", set()) if keyword in combined)
        return {"models": sorted(model_set), "attributes": attributes}

    def _doc_matches_attributes(self, doc: Document, attributes: list[str]) -> int:
        if not attributes:
            return 0
        text = "\n".join(
            [
                str(doc.metadata.get("heading_path") or ""),
                str(doc.metadata.get("doc_type") or ""),
                doc.page_content or "",
            ]
        )
        return sum(1 for attribute in attributes if attribute in text)

    def _evidence_priority(self, doc: Document, attributes: list[str]) -> tuple[int, int, int]:
        role_score = 2 if self._section_role(doc) == "evidence" else 0
        attribute_score = self._doc_matches_attributes(doc, attributes)
        doc_type = str(doc.metadata.get("doc_type") or "").lower()
        doc_type_score = {"specs": 3, "detail": 2, "manual": 2, "service": 1}.get(doc_type, 0)
        return role_score, attribute_score, doc_type_score

    def _complete_entity_coverage(
        self,
        docs: list[Document],
        *,
        query: str,
        normalized_query: str,
    ) -> tuple[list[Document], list[dict[str, Any]], dict[str, list[str]]]:
        if not getattr(self, "context_postprocess_enabled", True) or not getattr(self, "entity_coverage_enabled", True):
            return docs, [], {"models": [], "attributes": []}
        self._ensure_manifest_indexes()
        entities = self._extract_query_entities(query, normalized_query)
        required_models = entities["models"]
        attributes = entities["attributes"]
        if not required_models:
            return docs, [], entities
        covered_models = {str(doc.metadata.get("model") or "").upper() for doc in docs}
        existing_chunk_ids = {self._get_chunk_id(doc) for doc in docs}
        supplemental_docs: list[Document] = []
        supplemental_trace: list[dict[str, Any]] = []
        max_supplemental = getattr(self, "entity_coverage_max_supplemental_docs", 3)
        for model in required_models:
            if model in covered_models or len(supplemental_docs) >= max_supplemental:
                continue
            candidates = [
                doc
                for doc in (self._chunks_by_model or {}).get(model, [])
                if self._get_chunk_id(doc) not in existing_chunk_ids and self._section_role(doc) in {"evidence", "generic"}
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda doc: self._evidence_priority(doc, attributes), reverse=True)
            selected = candidates[0]
            supplemental_docs.append(selected)
            existing_chunk_ids.add(self._get_chunk_id(selected))
            supplemental_trace.append(
                {
                    "model": model,
                    "chunk_id": self._get_chunk_id(selected),
                    "heading_path": str(selected.metadata.get("heading_path", "")),
                    "doc_type": str(selected.metadata.get("doc_type", "")),
                    "reason": "entity_coverage",
                }
            )
        return [*docs, *supplemental_docs], supplemental_trace, entities

    def _expand_context_windows(self, docs: list[Document]) -> tuple[list[Document], list[dict[str, Any]]]:
        if not getattr(self, "context_postprocess_enabled", True):
            return docs, []
        self._ensure_manifest_indexes()
        expanded_docs: list[Document] = []
        expanded_trace: list[dict[str, Any]] = []
        max_expanded_docs = getattr(self, "context_window_max_expanded_docs", 4)
        target_chars = getattr(self, "context_window_target_chars", 650)
        neighbor_window = getattr(self, "context_window_neighbor_window", 1)
        for doc in docs[:max_expanded_docs]:
            source_path = str(doc.metadata.get("source_path") or doc.metadata.get("source") or "")
            source_docs = (self._chunks_by_source or {}).get(source_path, [])
            current_chunk_id = self._get_chunk_id(doc)
            current_index = next((index for index, candidate in enumerate(source_docs) if self._get_chunk_id(candidate) == current_chunk_id), None)
            if current_index is None:
                expanded_docs.append(doc)
                expanded_trace.append({"chunk_id": current_chunk_id, "expanded_chunk_ids": [current_chunk_id]})
                continue

            selected_indexes = {current_index}
            for distance in range(1, neighbor_window + 1):
                if current_index - distance >= 0:
                    selected_indexes.add(current_index - distance)
                if current_index + distance < len(source_docs):
                    selected_indexes.add(current_index + distance)
                total_chars = sum(len(source_docs[index].page_content or "") for index in selected_indexes)
                if total_chars >= target_chars:
                    break
            selected_docs = [source_docs[index] for index in sorted(selected_indexes)]
            expanded_content = "\n\n".join(doc.page_content or "" for doc in selected_docs if doc.page_content)
            metadata = dict(doc.metadata)
            metadata["expanded_from_chunk_id"] = current_chunk_id
            metadata["expanded_chunk_ids"] = json.dumps([self._get_chunk_id(selected_doc) for selected_doc in selected_docs], ensure_ascii=False)
            metadata["context_expanded"] = "true" if len(selected_docs) > 1 else "false"
            expanded_docs.append(Document(page_content=expanded_content, metadata=metadata))
            expanded_trace.append(
                {
                    "chunk_id": current_chunk_id,
                    "expanded_chunk_ids": [self._get_chunk_id(selected_doc) for selected_doc in selected_docs],
                    "expanded_char_count": len(expanded_content),
                }
            )
        if len(docs) > max_expanded_docs:
            expanded_docs.extend(docs[max_expanded_docs:])
        return expanded_docs, expanded_trace

    def _determine_query_bucket(self, query: str, normalized_query: str, detected_query_models: list[str]) -> str:
        if detected_query_models:
            return "explicit_model"
        combined = f"{query}\n{normalized_query}".lower()
        if self._weak_feature_pattern.search(combined):
            return "weak_feature"
        if any(keyword.lower() in combined for keyword in self._weak_feature_keywords):
            return "weak_feature"
        return "generic"

    def _is_manual_intent_query(self, query: str, normalized_query: str) -> bool:
        combined = f"{query}\n{normalized_query}".lower()
        return any(keyword.lower() in combined for keyword in self._manual_intent_keywords)

    def _get_weighted_rrf_v2_params(self, query_bucket: str) -> dict[str, float | int]:
        return dict(self.weighted_rrf_v2_bucket_conf.get(query_bucket, self.weighted_rrf_v2_bucket_conf["generic"]))

    def _weighted_rrf_v2_results(
        self,
        *,
        query_bucket: str,
        detected_query_models: list[str],
        manual_intent: bool,
        vector_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._get_weighted_rrf_v2_params(query_bucket)
        vector_weight = float(params["vector_weight"])
        bm25_weight = float(params["bm25_weight"])
        rrf_k = int(params["rrf_k"])
        fused_list = self._build_weighted_rrf_results(
            vector_results,
            bm25_results,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            rrf_k=rrf_k,
            candidate_limit=self.vector_top_k + self.bm25_top_k_v2,
        )
        detected_set = {model.upper() for model in detected_query_models}
        penalty_applied = False
        manual_bias_applied = False
        for item in fused_list:
            doc: Document = item["document"]
            model = str(doc.metadata.get("model", "")).strip().upper()
            doc_type = str(doc.metadata.get("doc_type", "")).strip().lower()
            adjusted = float(item.get("base_rrf_score") or 0.0)
            model_match = True
            model_penalty = False
            if query_bucket == "explicit_model" and detected_set and model and model != "SHARED" and model not in detected_set:
                adjusted *= self.model_mismatch_penalty
                model_match = False
                model_penalty = True
                penalty_applied = True
            if manual_intent and doc_type == "manual":
                adjusted *= self.manual_bias_boost
                item["manual_bias_applied"] = True
                manual_bias_applied = True
            item["model_match"] = model_match
            item["model_consistency_penalty_applied"] = model_penalty
            item["adjusted_rrf_score"] = adjusted
            item["rrf_score"] = adjusted
        fused_list.sort(key=lambda item: item.get("adjusted_rrf_score", 0.0), reverse=True)
        return fused_list, {
            "vector_weight": vector_weight,
            "bm25_weight": bm25_weight,
            "rrf_k": rrf_k,
            "model_consistency_penalty_applied": penalty_applied,
            "manual_bias_applied": manual_bias_applied,
        }

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

    def retrieve_docs(self, query: str, *, mode: str | None = None, rewrite: bool = True) -> list[Document]:
        actual_mode = mode or self.default_mode
        normalized_query = self.rewrite_query(query) if rewrite else query
        detected_query_models = self._extract_query_models(query, normalized_query)
        query_bucket = self._determine_query_bucket(query, normalized_query, detected_query_models)
        manual_intent = self._is_manual_intent_query(query, normalized_query)
        weighted_rrf_modes = {"weighted_rrf", "weighted_rrf_rerank", "weighted_rrf_v2", "weighted_rrf_v2_rerank"}
        weighted_rrf_v2_modes = {"weighted_rrf_v2", "weighted_rrf_v2_rerank"}
        bm25_top_k = self.bm25_top_k_v2 if actual_mode in weighted_rrf_v2_modes else self.bm25_top_k
        vector_results = self._vector_retrieve(normalized_query, top_k=self.vector_top_k) if actual_mode in {"vector", "hybrid", "hybrid_rerank", *weighted_rrf_modes} else []
        bm25_results = self._bm25_retrieve(normalized_query, top_k=bm25_top_k) if actual_mode in {"bm25", "hybrid", "hybrid_rerank", *weighted_rrf_modes} else []
        fusion_method = "hybrid_merge"
        applied_vector_weight: float | None = None
        applied_bm25_weight: float | None = None
        applied_rrf_k: int | None = None
        model_consistency_penalty_applied = False
        manual_bias_applied = False
        if actual_mode == "vector":
            merged_results = vector_results[: self.rerank_top_n]
            final_results = merged_results
        elif actual_mode == "bm25":
            merged_results = bm25_results[: self.rerank_top_n]
            final_results = merged_results
        elif actual_mode in weighted_rrf_v2_modes:
            fusion_method = "weighted_rrf_v2"
            merged_results, rrf_meta = self._weighted_rrf_v2_results(
                query_bucket=query_bucket,
                detected_query_models=detected_query_models,
                manual_intent=manual_intent,
                vector_results=vector_results,
                bm25_results=bm25_results,
            )
            applied_vector_weight = float(rrf_meta["vector_weight"])
            applied_bm25_weight = float(rrf_meta["bm25_weight"])
            applied_rrf_k = int(rrf_meta["rrf_k"])
            model_consistency_penalty_applied = bool(rrf_meta["model_consistency_penalty_applied"])
            manual_bias_applied = bool(rrf_meta["manual_bias_applied"])
            merged_results = self._apply_section_role_weighting(merged_results)
            final_results = self._rerank_results(
                query=query,
                normalized_query=normalized_query,
                merged_results=merged_results,
                enabled=(actual_mode == "weighted_rrf_v2_rerank"),
            )
        elif actual_mode in {"weighted_rrf", "weighted_rrf_rerank"}:
            fusion_method = "weighted_rrf"
            merged_results = self._weighted_rrf_results(vector_results, bm25_results)
            applied_vector_weight = self.vector_weight
            applied_bm25_weight = self.bm25_weight
            applied_rrf_k = self.rrf_k
            merged_results = self._apply_section_role_weighting(merged_results)
            final_results = self._rerank_results(
                query=query,
                normalized_query=normalized_query,
                merged_results=merged_results,
                enabled=(actual_mode == "weighted_rrf_rerank"),
            )
        else:
            merged_results = self._merge_results(vector_results, bm25_results)
            merged_results = self._apply_section_role_weighting(merged_results)
            final_results = self._rerank_results(
                query=query,
                normalized_query=normalized_query,
                merged_results=merged_results,
                enabled=(actual_mode == "hybrid_rerank"),
            )
        raw_docs = [item["document"] for item in final_results]
        completed_docs, supplemental_docs_trace, extracted_entities = self._complete_entity_coverage(
            raw_docs,
            query=query,
            normalized_query=normalized_query,
        )
        docs, expanded_docs_trace = self._expand_context_windows(completed_docs)
        retrieved_models = self._extract_retrieved_models([{"document": doc} for doc in completed_docs])
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
            "fusion_method": fusion_method,
            "query_bucket": query_bucket,
            "manual_intent": manual_intent,
            "vector_weight": applied_vector_weight,
            "bm25_weight": applied_bm25_weight,
            "rrf_k": applied_rrf_k,
            "applied_vector_weight": applied_vector_weight,
            "applied_bm25_weight": applied_bm25_weight,
            "detected_query_models": detected_query_models,
            "retrieved_models": retrieved_models,
            "model_confirmation_status": model_confirmation_status,
            "model_confirmation_source": model_confirmation_source,
            "should_reconfirm_model": should_reconfirm_model,
            "confirmed_model": confirmed_model,
            "model_consistency_penalty_applied": model_consistency_penalty_applied,
            "manual_bias_applied": manual_bias_applied,
            "vector_hit_count": len(vector_results),
            "bm25_hit_count": len(bm25_results),
            "merged_candidate_count": len(merged_results),
            "rerank_selected_count": len(final_results),
            "raw_doc_count": str(len(raw_docs)),
            "supplemental_doc_count": str(len(supplemental_docs_trace)),
            "doc_count": str(len(docs)),
            "context_postprocess_enabled": getattr(self, "context_postprocess_enabled", True),
            "extracted_entities": extracted_entities,
            "supplemental_docs": supplemental_docs_trace,
            "expanded_docs": expanded_docs_trace,
            "final_docs": [
                {
                    "chunk_id": str(item["document"].metadata.get("chunk_id", "")),
                    "source": item["source"],
                    "rank": index,
                    "doc_type": str(item["document"].metadata.get("doc_type", "")),
                    "model": str(item["document"].metadata.get("model", "")),
                    "score": item.get("score"),
                    "vector_rank": item.get("vector_rank"),
                    "bm25_rank": item.get("bm25_rank"),
                    "rrf_score": item.get("rrf_score"),
                    "base_rrf_score": item.get("base_rrf_score"),
                    "adjusted_rrf_score": item.get("adjusted_rrf_score"),
                    "section_role": item.get("section_role"),
                    "section_role_weight": item.get("section_role_weight"),
                    "model_match": item.get("model_match"),
                    "model_consistency_penalty_applied": item.get("model_consistency_penalty_applied", False),
                    "manual_bias_applied": item.get("manual_bias_applied", False),
                }
                for index, item in enumerate(final_results, start=1)
            ],
            "context_docs": [
                {
                    "chunk_id": str(doc.metadata.get("chunk_id", "")),
                    "rank": index,
                    "doc_type": str(doc.metadata.get("doc_type", "")),
                    "model": str(doc.metadata.get("model", "")),
                    "heading_path": str(doc.metadata.get("heading_path", "")),
                    "context_expanded": str(doc.metadata.get("context_expanded", "false")),
                    "expanded_from_chunk_id": str(doc.metadata.get("expanded_from_chunk_id", "")),
                }
                for index, doc in enumerate(docs, start=1)
            ],
        }
        if self.trace_callback is not None:
            self.trace_callback(query, docs)
        return docs

    def summarize_docs(self, query: str, docs: list[Document]) -> str:
        context_parts = []
        for index, doc in enumerate(docs, start=1):
            context_parts.append(f"【参考资料{index}】{doc.page_content} | 元数据: {doc.metadata}")
        return self.chain.invoke({"input": query, "context": "\n".join(context_parts)})

    def rag_summarize(self, query: str, *, mode: str | None = None) -> str:
        context_docs = self.retrieve_docs(query, mode=mode)
        return self.summarize_docs(query, context_docs)
