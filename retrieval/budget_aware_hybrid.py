from typing import List, Optional, Dict
import torch

from haystack.nodes.retriever import BaseRetriever
from haystack.nodes import BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.schema import Document, FilterType

from sentence_transformers import SentenceTransformer
from retrieval.scoring import score_documents


class BudgetAwareHybridRetriever(BaseRetriever):
    def __init__(
        self,
        *,
        document_store,
        embedding_model: str,
        token_budget: int,
        signal_weights: Dict[str, float],
        section_type_weights: Dict[str, float],
        use_gpu: bool = False,
        **kwargs,  # REQUIRED for Haystack 1.x
    ):
        # 🔑 Haystack 1.x: BaseRetriever takes NO args
        super().__init__(**kwargs)

        # 🔑 document_store is just an attribute in 1.x
        self.document_store = document_store
        self.store = document_store

        self.token_budget = token_budget
        self.signal_weights = signal_weights
        self.section_type_weights = section_type_weights

        # Dense encoder
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(embedding_model, device=device)

        # -------------------------
        # Separate BM25 store
        # -------------------------
        self.keyword_store = InMemoryDocumentStore(use_bm25=True)
        self.keyword_store.write_documents(
            document_store.get_all_documents()
        )

        self.bm25 = BM25Retriever(document_store=self.keyword_store)

    def retrieve(
        self,
        query: str,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        **kwargs,
    ) -> List[Document]:

        # -------------------------
        # Dense retrieval (FAISS)
        # -------------------------
        q_emb = self.encoder.encode([query])[0]
        dense_docs = self.store.query_by_embedding(
            query_emb=q_emb,
            top_k=50,
        )

        # -------------------------
        # Sparse retrieval (BM25)
        # -------------------------
        sparse_docs = self.bm25.retrieve(
            query=query,
            top_k=50,
        )

        # -------------------------
        # Merge dense + sparse
        # -------------------------
        merged = {}

        for d in dense_docs:
            merged[d.id] = {
                "doc": d,
                "semantic": d.score or 0.0,
                "lexical": 0.0,
            }

        for d in sparse_docs:
            merged.setdefault(
                d.id,
                {"doc": d, "semantic": 0.0, "lexical": 0.0}
            )
            merged[d.id]["lexical"] = d.score or 0.0

        # -------------------------
        # Multi-signal scoring
        # -------------------------
        scored_docs = score_documents(
            list(merged.values()),
            self.signal_weights,
            self.section_type_weights,
        )

        # -------------------------
        # Budget-aware selection
        # -------------------------
        return self._budget_select(scored_docs, top_k)

    def _budget_select(
        self,
        docs: List[Document],
        top_k: int,
    ) -> List[Document]:

        selected = []
        used_tokens = 0

        docs = sorted(
            docs,
            key=lambda d: d.score / max(d.meta.get("word_count", 1), 1),
            reverse=True,
        )

        for d in docs:
            wc = d.meta.get("word_count", 0)

            if used_tokens + wc <= self.token_budget:
                selected.append(d)
                used_tokens += wc

            if len(selected) >= top_k:
                break

        return sorted(selected, key=lambda d: d.score, reverse=True)

    def retrieve_batch(
        self,
        queries: List[str],
        filters: Optional[List[Optional[FilterType]]] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[List[Document]]:

        if top_k is None:
            top_k = 10

        if filters is None:
            filters = [None] * len(queries)

        return [
            self.retrieve(query=q, filters=f, top_k=top_k)
            for q, f in zip(queries, filters)
        ]
