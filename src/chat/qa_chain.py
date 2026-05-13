from __future__ import annotations

import uuid
from typing import Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.config import LLMConfig, RetrievalConfig
from src.models import AskResponse, ChatResponse, SourceRef
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever
from src.storage.vector_store import VectorStore

SYSTEM_PROMPT = """你是一个基于知识库的问答助手。请根据提供的上下文信息回答用户的问题。

要求：
1. 只基于提供的上下文内容回答，不要编造信息
2. 如果上下文中没有足够的信息，请明确告知
3. 用中文回答
4. 在回答中适当引用来源文件名"""


# 内存会话存储
_session_store: dict[str, list] = {}


def _get_session_history(session_id: str) -> list:
    """获取或创建会话历史。"""
    if session_id not in _session_store:
        _session_store[session_id] = []
    return _session_store[session_id]


def _format_docs(docs: list[Document]) -> str:
    """将检索到的文档格式化为上下文文本。"""
    return "\n\n".join(f"[来源: {doc.metadata.get('filename', '未知')}]\n{doc.page_content}" for doc in docs)


class QAChain:
    """问答链，支持单轮问答和多轮对话。"""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_config: LLMConfig,
        retrieval_config: RetrievalConfig,
    ) -> None:
        self._retriever = Retriever(vector_store)
        self._retrieval_config = retrieval_config
        self._vector_store = vector_store

        # 初始化重排序器（首次使用时延迟加载模型）
        self._reranker = (
            Reranker(model_name=retrieval_config.reranker_model)
            if retrieval_config.reranker_enabled
            else None
        )

        # 初始化 LLM
        llm_kwargs: dict = {
            "model": llm_config.model,
            "temperature": llm_config.temperature,
            "max_tokens": llm_config.max_tokens,
        }
        if llm_config.api_key:
            llm_kwargs["api_key"] = llm_config.api_key
        if llm_config.api_base:
            llm_kwargs["base_url"] = llm_config.api_base

        self._llm = ChatOpenAI(**llm_kwargs)

    # ── 单轮问答 ──────────────────────────────────────────

    def _retrieve_and_rerank(self, question: str, final_k: int) -> list[tuple[Document, float]]:
        """检索（初检 + 可选重排序），返回 (文档, 分数) 列表。"""
        fetch_k = self._retrieval_config.rerank_k if self._reranker else final_k
        results = self._retriever.retrieve(
            question,
            k=fetch_k,
            search_type=self._retrieval_config.search_type,
            score_threshold=self._retrieval_config.score_threshold,
            rrf_k=self._retrieval_config.rrf_k,
            hybrid_k=self._retrieval_config.hybrid_k,
        )
        if self._reranker:
            results = self._reranker.rerank(question, results, top_k=final_k)
        return results

    def ask(self, question: str) -> AskResponse:
        """单轮问答，不保留对话历史。"""
        scored_results = self._retrieve_and_rerank(question, self._retrieval_config.k)
        docs = [doc for doc, _ in scored_results]

        # 构建 prompt 并调用 LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "上下文信息：\n{context}\n\n问题：{question}"),
        ])
        chain = prompt | self._llm | StrOutputParser()

        answer = chain.invoke({
            "context": _format_docs(docs),
            "question": question,
        })

        # 来源引用
        sources = []
        for doc, score in scored_results:
            sources.append(SourceRef(
                document=doc.metadata.get("filename", "未知"),
                content=doc.page_content[:300],
                score=round(score, 4),
            ))

        return AskResponse(answer=answer, sources=sources)

    # ── 多轮对话 ──────────────────────────────────────────

    def chat(self, question: str, conversation_id: Optional[str] = None) -> ChatResponse:
        """多轮对话，保留对话历史。"""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        history = _get_session_history(conversation_id)
        history.append(HumanMessage(content=question))

        # 检索（初检 + 可选重排序）
        scored_results = self._retrieve_and_rerank(question, self._retrieval_config.k)
        docs = [doc for doc, _ in scored_results]

        # 构建带历史记录的 prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "上下文信息：\n{context}\n\n问题：{question}"),
        ])
        chain = prompt | self._llm | StrOutputParser()

        answer = chain.invoke({
            "chat_history": history,
            "context": _format_docs(docs),
            "question": question,
        })

        history.append(answer)

        # 来源引用
        sources = []
        for doc, score in scored_results:
            sources.append(SourceRef(
                document=doc.metadata.get("filename", "未知"),
                content=doc.page_content[:300],
                score=round(score, 4),
            ))

        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
        )
