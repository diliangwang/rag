from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class DocumentInfo(BaseModel):
    """文档元数据"""
    id: str
    filename: str
    source: str
    page_count: Optional[int] = None
    chunk_count: int = 0


class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    conversation_id: str


class SourceRef(BaseModel):
    """回答来源引用"""
    document: str
    content: str
    score: float
