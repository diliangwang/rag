from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document


def load_document(file_path: str) -> list[Document]:
    """根据文件扩展名自动选择合适的加载器加载文档。

    支持格式：.pdf, .txt, .md, .docx, .html
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        loader = Docx2txtLoader(str(path))
    elif suffix in (".txt", ".md"):
        loader = TextLoader(str(path), encoding="utf-8")
    elif suffix == ".html":
        loader = BSHTMLLoader(str(path), open_encoding="utf-8")
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

    docs = loader.load()

    # 为每个文档块添加元数据
    for doc in docs:
        doc.metadata["source"] = str(path)
        doc.metadata["filename"] = path.name

    return docs


def load_documents_from_directory(dir_path: str, supported_extensions: list[str] | None = None) -> list[Document]:
    """递归加载目录下所有受支持的文档。"""
    if supported_extensions is None:
        supported_extensions = [".pdf", ".txt", ".md", ".docx", ".html"]

    base = Path(dir_path)
    if not base.exists() or not base.is_dir():
        raise NotADirectoryError(f"目录不存在: {dir_path}")

    all_docs: list[Document] = []
    for ext in supported_extensions:
        for file_path in sorted(base.rglob(f"*{ext}")):
            if file_path.is_file():
                try:
                    docs = load_document(str(file_path))
                    all_docs.extend(docs)
                    print(f"  ✓ {file_path.name} ({len(docs)} 段)")
                except Exception as e:
                    print(f"  ✗ {file_path.name}: {e}")

    return all_docs
