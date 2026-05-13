from __future__ import annotations

import re
from typing import Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.storage.vector_store import VectorStore


class Retriever:
    """检索器，支持向量检索、MMR 和混合检索（向量 + BM25 关键词）。"""

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: list[Document] = []

    # ── BM25 索引构建 ──────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """将文本分词为词元列表（用于 BM25）。"""
        return re.findall(r"\w+", text.lower())

    def rebuild_bm25_index(self) -> None:
        """从向量库中重建 BM25 索引（导入/删除文档后需调用以保持同步）。"""
        contents = self._vector_store.get_all_contents()
        corpus: list[list[str]] = []
        self._bm25_docs = []
        for content, metadata in contents:
            if content:
                corpus.append(self._tokenize(content))
                self._bm25_docs.append(Document(page_content=content, metadata=metadata))
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def _ensure_bm25(self) -> None:
        """确保 BM25 索引已构建（懒初始化）。"""
        if self._bm25 is None:
            self.rebuild_bm25_index()

    def _bm25_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """BM25 关键词搜索，返回 (文档, 分数) 列表。"""
        self._ensure_bm25()
        if self._bm25 is None or not self._bm25_docs:
            return []
        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [
            (self._bm25_docs[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

    # ── RRF 融合 ───────────────────────────────────────────

    @staticmethod
    def _rrf_merge(
        vector_results: list[tuple[Document, float]],
        keyword_results: list[tuple[Document, float]],
        k: int,
        rrf_k: int = 60,
    ) -> list[tuple[Document, float]]:
        """用倒数排序融合（Reciprocal Rank Fusion）合并向量 + 关键词结果。"""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        def _key(doc: Document) -> str:
            # 用内容前 200 字符的 hash 作为文档标识
            return str(hash(doc.page_content[:200]))

        for rank, (doc, _) in enumerate(vector_results):
            key = _key(doc)
            scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
            doc_map[key] = doc

        for rank, (doc, _) in enumerate(keyword_results):
            key = _key(doc)
            scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
            doc_map.setdefault(key, doc)

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(doc_map[key], score) for key, score in ranked[:k]]

    # ── 主检索入口 ─────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = 4,
        search_type: str = "similarity",
        score_threshold: float = 0.0,
        rrf_k: int = 60,
        hybrid_k: int = 4,
    ) -> list[tuple[Document, float]]:
        """执行检索，返回 (文档, 分数) 列表。

        Args:
            query: 查询文本。
            k: 返回的文档数量。
            search_type: similarity / mmr / hybrid。
            score_threshold: 向量检索的相似度阈值。
            rrf_k: 混合检索中 RRF 常数。
            hybrid_k: 混合检索中 BM25 关键词检索的文档数。
        """
        if search_type == "mmr":
            docs = self._vector_store.mmr_search(query, k=k)
            return [(doc, 0.0) for doc in docs]

        if search_type == "hybrid":
            vector_results = self._vector_store.similarity_search(
                query, k=k, score_threshold=score_threshold,
            )
            keyword_results = self._bm25_search(query, k=hybrid_k)
            return self._rrf_merge(vector_results, keyword_results, k, rrf_k=rrf_k)

        # 默认：相似度搜索
        return self._vector_store.similarity_search(
            query, k=k, score_threshold=score_threshold,
        )

    def format_sources(self, results: list[tuple[Document, float]]) -> list[dict]:
        """将检索结果格式化为来源引用列表。"""
        sources = []
        for doc, score in results:
            sources.append({
                "document": doc.metadata.get("filename", doc.metadata.get("source", "未知")),
                "content": doc.page_content[:300],
                "score": round(score, 4),
            })
        return sources
