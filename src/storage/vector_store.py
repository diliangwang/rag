from __future__ import annotations

from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import VectorStoreConfig


class VectorStore:
    """向量存储管理器，封装 ChromaDB 操作。"""

    def __init__(
        self,
        embeddings: Embeddings,
        config: VectorStoreConfig,
    ) -> None:
        self._embeddings = embeddings
        self._config = config
        self._collection_name = config.collection_name
        self._persist_directory = config.persist_directory

        self._db = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=self._persist_directory,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """添加文档到向量库，返回文档 ID 列表。"""
        return self._db.add_documents(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.0,
    ) -> list[tuple[Document, float]]:
        """相似度搜索，返回 (文档, 分数) 列表。"""
        return self._db.similarity_search_with_relevance_scores(
            query,
            k=k,
            score_threshold=score_threshold,
        )

    def mmr_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> list[Document]:
        """MMR 搜索，平衡相关性和多样性。"""
        return self._db.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

    def get_all_documents(self) -> list[dict]:
        """列出所有已导入的文档元数据（包含每个源文件的块数）。"""
        all_docs = self._db.get(include=["metadatas"])
        sources: dict[str, dict] = {}
        for doc_id, metadata in zip(all_docs["ids"], all_docs["metadatas"]):
            if metadata and (source := metadata.get("source")):
                if source not in sources:
                    sources[source] = {
                        "id": doc_id,
                        "source": source,
                        "filename": metadata.get("filename", source),
                        "chunk_count": 0,
                    }
                sources[source]["chunk_count"] += 1
        return list(sources.values())

    def get_chunks_by_source(self, source: str) -> list[dict]:
        """获取指定源文件的所有文档块内容。"""
        result = self._db.get(where={"source": source}, include=["metadatas", "documents"])
        chunks = []
        for doc_id, doc_content, metadata in zip(
            result["ids"], result["documents"], result["metadatas"]
        ):
            chunks.append({
                "id": doc_id,
                "content": doc_content,
                "metadata": metadata,
            })
        return chunks

    def get_all_contents(self) -> list[tuple[str, dict]]:
        """获取向量库中所有文档的文本内容和元数据（用于构建 BM25 索引）。"""
        result = self._db.get(include=["documents", "metadatas"])
        return list(zip(result["documents"], result["metadatas"]))

    def delete_document(self, doc_id: str) -> None:
        """删除指定 ID 的文档。"""
        self._db.delete(ids=[doc_id])

    def delete_by_source(self, source: str) -> int:
        """删除指定源文件的所有文档块，返回删除的文档块数量。"""
        all_docs = self._db.get(where={"source": source})
        ids_to_delete = list(all_docs["ids"])
        if ids_to_delete:
            self._db.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    def count(self) -> int:
        """返回向量库中的文档块数量。"""
        return self._db._collection.count()

    def as_retriever(self, search_type: str = "similarity", k: int = 4):
        """获取 LangChain Retriever 对象。"""
        return self._db.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k},
        )
