"""重排序器

使用 Cross-Encoder 模型对初检结果进行重排序，提高检索精度。
依赖 sentence-transformers（可选），可通过 `uv sync --extra reranker` 安装。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document


class Reranker:
    """Cross-Encoder 重排序器。

    对 (query, doc) 逐对计算相关性分数，替代向量相似度评分。
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self) -> None:
        """延迟加载 Cross-Encoder 模型。"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装，请执行：uv sync --extra reranker"
            )

    def rerank(
        self,
        query: str,
        documents: list[tuple[Document, float]],
        top_k: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """对检索结果进行重排序。

        Args:
            query: 用户查询。
            documents: 初检结果列表，每项为 (文档, 原始分数)。
            top_k: 返回前 k 条结果。默认返回全部。

        Returns:
            按 Cross-Encoder 分数降序排列的 (文档, 重排序分数) 列表。
        """
        if not documents:
            return []

        self._load_model()

        pairs = [(query, doc.page_content) for doc, _ in documents]
        scores = self._model.predict(pairs)

        ranked = sorted(
            [(doc, float(score)) for (doc, _), score in zip(documents, scores)],
            key=lambda x: -x[1],
        )
        return ranked[:top_k] if top_k else ranked
