from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from openai import OpenAIError
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.chat.qa_chain import QAChain
from src.config import Settings
from src.embedding.provider import create_embeddings
from src.ingestion.loader import load_document
from src.ingestion.splitter import split_documents
from src.models import AskRequest, AskResponse, ChatRequest, ChatResponse
from src.storage.vector_store import VectorStore

app = FastAPI(
    title="RAG API",
    description="基于 LangChain 的通用文档问答 RAG 系统",
    version="0.1.0",
)

_settings: Settings | None = None
_store: VectorStore | None = None
_qa: QAChain | None = None
_cors_configured: bool = False


def _configure_cors(settings: Settings) -> None:
    """根据配置注册 CORS 中间件。"""
    global _cors_configured
    if _cors_configured:
        return
    cors_config = settings.server.cors
    if not cors_config.enabled:
        _cors_configured = True
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.allow_origins,
        allow_credentials=cors_config.allow_credentials,
        allow_methods=cors_config.allow_methods,
        allow_headers=cors_config.allow_headers,
    )
    _cors_configured = True


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml("config.yaml")
    return _settings


# 模块导入时注册 CORS 中间件
_configure_cors(get_settings())


def get_store() -> VectorStore:
    global _store
    if _store is None:
        settings = get_settings()
        embeddings = create_embeddings(settings.embedding)
        _store = VectorStore(embeddings, settings.vector_store)
    return _store


def get_qa() -> QAChain:
    global _qa
    if _qa is None:
        settings = get_settings()
        store = get_store()
        _qa = QAChain(store, settings.llm, settings.retrieval)
    return _qa


@app.exception_handler(OpenAIError)
async def openai_error_handler(request: Request, exc: OpenAIError):
    return JSONResponse(
        status_code=503,
        content={"detail": f"LLM 服务不可用: {exc}"},
    )


@app.get("/health")
async def health():
    """健康检查。"""
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
):
    """上传并导入文档。"""
    settings = get_settings()
    store = get_store()

    # 验证文件扩展名
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.ingestion.supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {suffix}，支持: {settings.ingestion.supported_extensions}",
        )

    # 保存临时文件
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        docs = load_document(tmp_path)
        chunks = split_documents(
            docs,
            chunk_size=chunk_size or settings.ingestion.chunk_size,
            chunk_overlap=chunk_overlap or settings.ingestion.chunk_overlap,
        )
        store.add_documents(chunks)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return JSONResponse({
        "message": f"成功导入 {len(chunks)} 个文档块",
        "filename": file.filename,
        "chunks": len(chunks),
    })


@app.get("/documents")
async def list_documents():
    """列出所有文档。"""
    store = get_store()
    docs = store.get_all_documents()
    total = store.count()
    return {"total_chunks": total, "documents": docs}


@app.delete("/documents/{source:path}")
async def delete_document(source: str):
    """删除指定文档。"""
    store = get_store()
    count = store.delete_by_source(source)
    if count == 0:
        raise HTTPException(status_code=404, detail=f"未找到文档: {source}")
    return {"message": f"已删除 {count} 个文档块", "source": source, "deleted": count}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """单轮问答。"""
    qa = get_qa()
    store = get_store()
    if store.count() == 0:
        raise HTTPException(status_code=400, detail="知识库为空，请先导入文档")
    return qa.ask(request.question)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """多轮对话。"""
    qa = get_qa()
    store = get_store()
    if store.count() == 0:
        raise HTTPException(status_code=400, detail="知识库为空，请先导入文档")
    return qa.chat(request.question, request.conversation_id)
