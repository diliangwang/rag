from __future__ import annotations

from typing import Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.semantic_chunker import semantic_split_documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """将文档分割成更小的块（基于固定长度）。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.split_documents(documents)


def semantic_split(
    documents: list[Document],
    embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    window_size: int = 3,
    threshold_percentile: float = 60.0,
    min_chunk_chars: int = 100,
    max_chunk_chars: int = 2000,
) -> list[Document]:
    """将文档按语义相似度分割成语义连贯的块。

    Args:
        documents: 要分割的文档列表。
        embed_fn: 嵌入函数。
        window_size: 语义窗口大小（句子数）。
        threshold_percentile: 断点阈值百分位。
        min_chunk_chars: 最小块字符数。
        max_chunk_chars: 最大块字符数。

    Returns:
        语义分割后的文档列表。
    """
    return semantic_split_documents(
        documents,
        embed_fn,
        window_size=window_size,
        threshold_percentile=threshold_percentile,
        min_chunk_chars=min_chunk_chars,
        max_chunk_chars=max_chunk_chars,
    )
