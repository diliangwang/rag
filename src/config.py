from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml


class LLMConfig(BaseSettings):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    temperature: float = 0.0
    max_tokens: int = 2048

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
    )


class EmbeddingConfig(BaseSettings):
    provider: str = "openai"
    model: str = "text-embedding-ada-002"
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        extra="ignore",
    )


class VectorStoreConfig(BaseSettings):
    persist_directory: str = "data/chroma_db"
    collection_name: str = "rag_docs"

    model_config = SettingsConfigDict(env_prefix="VECTOR_STORE_", extra="ignore")


class SemanticConfig(BaseSettings):
    window_size: int = 3
    threshold_percentile: float = 60.0
    min_chunk_chars: int = 100
    max_chunk_chars: int = 2000

    model_config = SettingsConfigDict(extra="ignore")


class IngestionConfig(BaseSettings):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_mode: Literal["fixed", "semantic"] = "semantic"
    supported_extensions: list[str] = [".pdf", ".txt", ".md", ".docx", ".html"]
    semantic: SemanticConfig = SemanticConfig()

    model_config = SettingsConfigDict(env_prefix="INGESTION_", extra="ignore")


class RetrievalConfig(BaseSettings):
    k: int = 4
    score_threshold: float = 0.0
    search_type: Literal["similarity", "mmr", "hybrid"] = "similarity"
    rrf_k: int = 60          # RRF 常数，越小排名靠前的文档权重越大
    hybrid_k: int = 4         # 混合检索中 BM25 关键词检索的文档数
    reranker_enabled: bool = False  # 是否启用 Cross-Encoder 重排序
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-Encoder 模型
    rerank_k: int = 20       # 重排序前初检的文档数（值越大效果越好但越慢）

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", extra="ignore")


class ServerConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False

    model_config = SettingsConfigDict(env_prefix="SERVER_", extra="ignore")


class Settings(BaseSettings):
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    ingestion: IngestionConfig = IngestionConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    server: ServerConfig = ServerConfig()

    model_config = SettingsConfigDict(extra="ignore")

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> Settings:
        """从 YAML 文件加载配置，支持环境变量覆盖。"""
        # 加载 .env 文件到环境变量，使 os.getenv 能读取到
        load_dotenv()

        yaml_path = Path(path)
        if not yaml_path.exists():
            return cls()

        with open(yaml_path) as f:
            raw = yaml.safe_load(f) or {}

        llm = raw.get("llm", {})
        embedding = raw.get("embedding", {})
        vs = raw.get("vector_store", {})
        ingestion = raw.get("ingestion", {})
        retrieval = raw.get("retrieval", {})
        server = raw.get("server", {})

        # 环境变量优先级高于 YAML
        if api_key := os.getenv("LLM_API_KEY"):
            llm["api_key"] = api_key
        if api_base := os.getenv("LLM_API_BASE"):
            llm["api_base"] = api_base
        if model := os.getenv("LLM_MODEL"):
            llm["model"] = model

        if api_key := os.getenv("EMBEDDING_API_KEY"):
            embedding["api_key"] = api_key
        if api_base := os.getenv("EMBEDDING_API_BASE"):
            embedding["api_base"] = api_base
        if model := os.getenv("EMBEDDING_MODEL"):
            embedding["model"] = model

        return cls(
            llm=LLMConfig(**llm),
            embedding=EmbeddingConfig(**embedding),
            vector_store=VectorStoreConfig(**vs),
            ingestion=IngestionConfig(**ingestion),
            retrieval=RetrievalConfig(**retrieval),
            server=ServerConfig(**server),
        )
