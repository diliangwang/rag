"""语义分块器

根据文本语义相似度自动识别主题边界，将文档分割为语义连贯的文本块。
思路：将文本拆分为句子后分组嵌入，相邻组间相似度显著下降的位置即为断点。
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
from langchain_core.documents import Document

# 段落分隔符（连续换行）
_PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")


def _split_sentences(text: str) -> list[str]:
    """将文本拆分为句子列表，保留完整句子边界。"""
    paragraphs = _PARAGRAPH_PATTERN.split(text.strip())
    sentences: list[str] = []

    # 段落合并阈值（无句号中断时强行分割的长度）
    MAX_SENTENCE_CHARS = 200

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 用句尾标点分割段落，同时保留标点
        # split 加捕获组会使标点保留在结果中
        parts = re.split(r"([。！？：.!?\n]+)", para)
        buffer = ""
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            if not part:
                i += 1
                continue

            buffer += part

            # 如果有下一个元素且它是标点，追加到 buffer 并作为一个完整句子
            if i + 1 < len(parts) and parts[i + 1].strip():
                punct = parts[i + 1].strip()
                buffer += punct
                sentences.append(buffer)
                buffer = ""
                i += 2  # 跳过标点位
            elif len(buffer) >= MAX_SENTENCE_CHARS:
                # 没有标点但 buffer 过长，强制分割
                sentences.append(buffer)
                buffer = ""
                i += 1
            else:
                i += 1

        if buffer:
            sentences.append(buffer)

    return [s for s in sentences if len(s.strip()) >= 2]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def semantic_chunk_text(
    text: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    window_size: int = 3,
    threshold_percentile: float = 60.0,
    min_chunk_chars: int = 100,
    max_chunk_chars: int = 2000,
) -> list[str]:
    """将文本按语义相似度分割为多个块。

    Args:
        text: 输入文本。
        embed_fn: 嵌入函数，接受字符串列表，返回向量列表。
        window_size: 每个滑动窗口包含的句子数。
        threshold_percentile: 断点阈值百分位（相似度低于此百分位的视为断点）。
        min_chunk_chars: 最小块字符数（低于此值会与相邻块合并）。
        max_chunk_chars: 最大块字符数（超过此值会强制截断）。

    Returns:
        语义分割后的文本块列表。
    """
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text]

    # 将句子分组为窗口（每组 window_size 个句子）
    windows: list[str] = []
    window_indices: list[tuple[int, int]] = []  # (start_idx, end_idx) in sentences

    for i in range(0, len(sentences), window_size):
        group = sentences[i : i + window_size]
        windows.append("".join(group))
        window_indices.append((i, i + len(group)))

    if len(windows) <= 1:
        return [text]

    # 计算每个窗口的嵌入向量
    try:
        embeddings = embed_fn(windows)
    except Exception:
        # 嵌入失败时回退到段落级拆分
        return _paragraph_fallback(text, max_chunk_chars)

    if not embeddings or len(embeddings) != len(windows):
        return _paragraph_fallback(text, max_chunk_chars)

    # 计算相邻窗口间的余弦相似度
    similarities: list[float] = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    if not similarities:
        return [text]

    # 确定阈值：低于 threshold_percentile 分位数的视为断点
    threshold = float(np.percentile(similarities, threshold_percentile))

    # 找出断点位置
    breakpoints: list[int] = []
    for i, sim in enumerate(similarities):
        if sim < threshold:
            # 断点在 window_indices[i] 的末尾
            bp = window_indices[i][1]
            if bp > 0 and bp < len(sentences):
                breakpoints.append(bp)

    # 按断点合并句子为块
    chunks: list[str] = []
    prev = 0
    for bp in sorted(set(breakpoints)):
        chunk_text = "".join(sentences[prev:bp]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        prev = bp

    # 最后一块
    remaining = "".join(sentences[prev:]).strip()
    if remaining:
        chunks.append(remaining)

    # 如果没有产生断点，整篇作为一个块
    if len(chunks) <= 1:
        return _paragraph_fallback(text, max_chunk_chars)

    # 后处理：合并过小的块，分割过大的块
    chunks = _merge_small_chunks(chunks, min_chunk_chars)
    chunks = _split_large_chunks(chunks, max_chunk_chars)

    return [c for c in chunks if c.strip()]


def _paragraph_fallback(text: str, max_chunk_chars: int) -> list[str]:
    """回退策略：按段落分割。"""
    paragraphs = _PARAGRAPH_PATTERN.split(text.strip())
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buffer) + len(para) > max_chunk_chars and buffer:
            chunks.append(buffer)
            buffer = para
        else:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para
    if buffer:
        chunks.append(buffer)
    return chunks or [text]


def _merge_small_chunks(chunks: list[str], min_chars: int) -> list[str]:
    """将过小的块合并到相邻块。"""
    result: list[str] = []
    for chunk in chunks:
        if result and len(chunk) < min_chars:
            result[-1] = result[-1] + "\n\n" + chunk
        else:
            result.append(chunk)
    return result


def _split_large_chunks(chunks: list[str], max_chars: int) -> list[str]:
    """将过大的块按句子边界截断。"""
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            result.append(chunk)
        else:
            # 在 max_chars 附近找句子边界截断
            parts = _PARAGRAPH_PATTERN.split(chunk)
            buffer = ""
            for part in parts:
                if len(buffer) + len(part) > max_chars and buffer:
                    result.append(buffer)
                    buffer = part
                else:
                    buffer = (buffer + "\n\n" + part).strip() if buffer else part
            if buffer:
                result.append(buffer)
    return result


def semantic_split_documents(
    documents: list[Document],
    embed_fn: Callable[[list[str]], list[list[float]]],
    *,
    window_size: int = 3,
    threshold_percentile: float = 60.0,
    min_chunk_chars: int = 100,
    max_chunk_chars: int = 2000,
) -> list[Document]:
    """对文档列表执行语义分割，保留元数据。"""
    result_docs: list[Document] = []
    for doc in documents:
        chunks = semantic_chunk_text(
            doc.page_content,
            embed_fn,
            window_size=window_size,
            threshold_percentile=threshold_percentile,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars,
        )
        for chunk in chunks:
            result_docs.append(Document(page_content=chunk, metadata=dict(doc.metadata)))
    return result_docs
