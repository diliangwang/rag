from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from src.config import EmbeddingConfig


def create_embeddings(config: EmbeddingConfig) -> OpenAIEmbeddings:
    """根据配置创建 Embedding 模型实例。

    支持 OpenAI 及所有 OpenAI-compatible API（如 SiliconFlow）。
    """
    kwargs = {
        "model": config.model,
        # DashScope 限制每批最多 10 条，OpenAI 限制 2048 条
        "chunk_size": 10,
        # 关闭 tiktoken 本地分词，确保发送原始文本而非 token ID
        # 部分 OpenAI-compatible API（如阿里云 DashScope）不支持 token ID 格式
        "tiktoken_enabled": False,
        "check_embedding_ctx_length": False,
    }

    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.api_base:
        kwargs["openai_api_base"] = config.api_base

    return OpenAIEmbeddings(**kwargs)
