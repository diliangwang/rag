from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openai import OpenAIError

from src.config import Settings
from src.embedding.provider import create_embeddings
from src.ingestion.loader import load_document, load_documents_from_directory
from src.ingestion.splitter import semantic_split, split_documents
from src.storage.vector_store import VectorStore

app = typer.Typer(
    name="rag",
    help="RAG 系统 - 基于 LangChain 的通用文档问答",
    no_args_is_help=True,
)
console = Console()

_global_settings: Settings | None = None
_global_store: VectorStore | None = None


def handle_error(e: Exception) -> None:
    """统一错误处理."""
    if isinstance(e, OpenAIError):
        console.print(f"[red]API 密钥错误:[/red] {e}")
        console.print("[yellow]请设置环境变量 LLM_API_KEY 和 EMBEDDING_API_KEY，或创建 .env 文件[/yellow]")
    elif isinstance(e, FileNotFoundError):
        console.print(f"[red]文件未找到:[/red] {e}")
    else:
        console.print(f"[red]错误:[/red] {e}")


def get_settings() -> Settings:
    global _global_settings
    if _global_settings is None:
        _global_settings = Settings.from_yaml("config.yaml")
    return _global_settings


def get_store() -> VectorStore:
    global _global_store
    if _global_store is None:
        settings = get_settings()
        embeddings = create_embeddings(settings.embedding)
        _global_store = VectorStore(embeddings, settings.vector_store)
    return _global_store


@app.command()
def ingest(
    path: str = typer.Argument(..., help="文件或目录路径"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="分块大小"),
    chunk_overlap: Optional[int] = typer.Option(None, "--chunk-overlap", help="分块重叠"),
) -> None:
    """导入文档到知识库。"""
    settings = get_settings()
    store = get_store()
    input_path = Path(path)

    console.print(f"[bold]导入文档:[/bold] {path}")

    if input_path.is_file():
        docs = load_document(path)
    elif input_path.is_dir():
        docs = load_documents_from_directory(path, settings.ingestion.supported_extensions)
    else:
        console.print(f"[red]路径无效: {path}[/red]")
        raise typer.Exit(1)

    if not docs:
        console.print("[yellow]未找到可导入的文档[/yellow]")
        return

    console.print(f"\n[bold]分割文档...[/bold] ({len(docs)} 段 → ", end="")

    if settings.ingestion.chunk_mode == "semantic":
        emb_fn = create_embeddings(settings.embedding).embed_documents
        sem_cfg = settings.ingestion.semantic
        chunks = semantic_split(
            docs,
            emb_fn,
            window_size=sem_cfg.window_size,
            threshold_percentile=sem_cfg.threshold_percentile,
            min_chunk_chars=sem_cfg.min_chunk_chars,
            max_chunk_chars=sem_cfg.max_chunk_chars,
        )
        console.print(f"{len(chunks)} 块, [cyan]语义模式[/cyan])")
    else:
        chunks = split_documents(
            docs,
            chunk_size=chunk_size or settings.ingestion.chunk_size,
            chunk_overlap=chunk_overlap or settings.ingestion.chunk_overlap,
        )
        console.print(f"{len(chunks)} 块, [dim]固定长度模式[/dim])")

    console.print("[bold]写入向量库...[/bold]")
    store.add_documents(chunks)
    console.print(f"[green]✓ 成功导入 {len(chunks)} 个文档块[/green]")


@app.command()
def list_docs() -> None:
    """列出知识库中的所有文档。"""
    store = get_store()
    documents = store.get_all_documents()
    total_chunks = store.count()

    if not documents:
        console.print("[yellow]知识库为空，请先使用 rag ingest 导入文档[/yellow]")
        return

    table = Table(title=f"📚 知识库文档 (共 {total_chunks} 个文档块)")
    table.add_column("文件名", style="cyan")
    table.add_column("来源路径", style="green")
    table.add_column("文档块数", style="yellow")

    for doc in documents:
        table.add_row(doc["filename"], doc["source"], str(doc["chunk_count"]))

    console.print(table)


@app.command()
def delete(
    source: str = typer.Argument(..., help="要删除的文档源路径"),
) -> None:
    """删除指定文档。"""
    store = get_store()
    count = store.delete_by_source(source)
    if count > 0:
        console.print(f"[green]✓ 已删除 {count} 个文档块: {source}[/green]")
    else:
        console.print(f"[yellow]未找到匹配的文档: {source}[/yellow]")
        console.print("[dim]提示: 使用 rag list-docs 查看正确的源路径[/dim]")


@app.command()
def chunks(
    source: str = typer.Argument(..., help="文档源路径（来自 rag list-docs）"),
    show_all: bool = typer.Option(False, "--all", "-a", help="显示每块的完整内容"),
) -> None:
    """查看指定文档的每个块的内容。"""
    store = get_store()
    chunks_list = store.get_chunks_by_source(source)

    if not chunks_list:
        console.print(f"[yellow]未找到文档: {source}[/yellow]")
        console.print("[dim]提示: 使用 rag list-docs 查看可用的文档路径[/dim]")
        return

    console.print(f"[bold]文档:[/bold] {source}")
    console.print(f"[dim]共 {len(chunks_list)} 个文档块[/dim]\n")

    for i, chunk in enumerate(chunks_list, 1):
        content = chunk["content"] if show_all else chunk["content"][:300]
        suffix = "..." if not show_all and len(chunk["content"]) > 300 else ""
        metadata = chunk["metadata"] or {}

        panel = Panel(
            f"{content}{suffix}",
            title=f"[bold]块 {i}/{len(chunks_list)}[/bold]  [dim]id: {chunk['id'][:8]}...[/dim]",
            border_style="blue",
        )
        console.print(panel)


@app.command()
def ask(
    question: str = typer.Argument(..., help="你的问题"),
) -> None:
    """提问并获取回答（单轮问答）。"""
    from src.chat.qa_chain import QAChain

    settings = get_settings()
    store = get_store()

    if store.count() == 0:
        console.print("[yellow]知识库为空，请先导入文档[/yellow]")
        raise typer.Exit(1)

    qa = QAChain(store, settings.llm, settings.retrieval)

    with console.status("[bold green]思考中...[/bold green]"):
        response = qa.ask(question)

    console.print(Panel(response.answer, title="[bold]回答[/bold]", border_style="green"))

    if response.sources:
        console.print("\n[bold]来源:[/bold]")
        for src in response.sources:
            console.print(f"  [cyan]{src.document}[/cyan] (相似度: {src.score:.2f})")
            console.print(f"  {src.content[:150]}...")
            console.print()


@app.command()
def chat() -> None:
    """进入交互式对话模式（连续多轮问答）。"""
    from src.chat.qa_chain import QAChain

    settings = get_settings()
    store = get_store()

    if store.count() == 0:
        console.print("[yellow]知识库为空，请先导入文档[/yellow]")
        raise typer.Exit(1)

    qa = QAChain(store, settings.llm, settings.retrieval)
    conversation_id: str | None = None

    console.print("[bold cyan]RAG 交互式对话[/bold cyan] (输入 'exit' 退出, 'clear' 清空历史)")
    console.print()

    while True:
        question = console.input("[bold]问题:[/bold] ")
        if question.lower() in ("exit", "quit"):
            break
        if question.lower() == "clear":
            conversation_id = None
            console.print("[yellow]已清空对话历史[/yellow]")
            continue
        if not question.strip():
            continue

        with console.status("[bold green]思考中...[/bold green]"):
            response = qa.chat(question, conversation_id)
        conversation_id = response.conversation_id

        console.print(Panel(response.answer, title="[bold]回答[/bold]", border_style="green"))

        if response.sources:
            console.print("[dim]来源: " + ", ".join(s.document for s in response.sources) + "[/dim]")
        console.print()


@app.command()
def serve(
    host: Optional[str] = typer.Option(None, "--host", help="监听地址"),
    port: Optional[int] = typer.Option(None, "--port", help="监听端口"),
) -> None:
    """启动 API 服务。"""
    import uvicorn

    settings = get_settings()
    host = host or settings.server.host
    port = port or settings.server.port

    console.print(f"[bold]启动 API 服务:[/bold] http://{host}:{port}")
    console.print("[dim]API 文档: http://{0}:{1}/docs[/dim]".format(host, port))

    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=settings.server.reload,
    )


def cli_main():
    """Entry point for console_scripts with global error handling."""
    try:
        app()
    except OpenAIError as e:
        handle_error(e)
        raise SystemExit(1)
    except Exception as e:
        handle_error(e)
        raise SystemExit(1)


if __name__ == "__main__":
    app()
