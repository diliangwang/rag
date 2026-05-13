# CLAUDE.md

此文件为 Claude Code 提供本仓库的上下文和开发指南。

## 项目概述

基于 LangChain + ChromaDB 的通用文档问答 RAG 系统。支持 CLI 和 API 双模式，灵活切换 LLM 提供商，支持固定长度和语义两种分块模式，检索支持向量/MMR/混合（向量+BM25）三种方式。

技术栈：Python 3.12 / uv / LangChain 0.3+ / ChromaDB / FastAPI / Typer

## 常用命令

### 应用命令
```bash
uv run rag ingest <path>     # 导入文档（文件或目录）
uv run rag list-docs         # 列出文档及块数
uv run rag delete <source>   # 按来源路径删除文档
uv run rag chunks <source>   # 查看文档块内容（-a 显示全文）
uv run rag ask <question>    # 单轮问答
uv run rag chat              # 交互式多轮对话
uv run rag serve             # 启动 API 服务（默认 :8000）
uv run python -m src         # 同 `rag --help`
```

### 开发命令
```bash
uv sync                      # 安装/同步依赖
uv lock                      # 锁定依赖版本（修改 pyproject.toml 后执行）
uv run python3 -c "..."      # 快速内联测试
```

添加依赖：在 `pyproject.toml` 的 `[project.dependencies]` 中添加，然后执行 `uv sync`。
可选依赖（重排序器）需 `uv sync --extra reranker` 安装。

## 工程规范

- **`from __future__ import annotations`** — 每个 `.py` 文件以该行开头
- **完整类型标注** — 所有函数签名标注类型，使用 `Optional[X]`（不用 `X | None`，在 `from __future__ import annotations` 下确保兼容性）
- **统一 OpenAI 接口** — LLM 通过 `ChatOpenAI`、Embedding 通过 `OpenAIEmbeddings` 调用。切换提供商只需改 `api_base` + `model`
- **入口点** — `entry.py` → `main.cli_main`（`rag` console_scripts），`__main__.py` → `main.app()`（`python -m src`）
- **全局错误处理** — `main.py:handle_error()` 统一处理 OpenAIError（API 密钥提示）和 FileNotFoundError（友好提示）。CLI 入口 try/except 包裹后 `raise SystemExit(1)`
- **文档元数据** — 每个加载的文档添加 `source`（绝对路径）和 `filename`（文件名）两个元数据字段
- **配置优先级** — `.env` > `config.yaml` > pydantic 默认值。`Settings.from_yaml()` 在读取 YAML 之前先调用 `load_dotenv()`

## 架构全景

```
src/
├── entry.py            # console_scripts 入口 → cli_main()
├── __main__.py         # python -m src 入口 → app()
├── main.py             # Typer CLI：ingest / list-docs / delete / chunks / ask / chat / serve
├── api.py              # FastAPI：health / ingest / documents / delete / ask / chat
│                       #   （通过全局单例与 main.py 共享 Settings/VectorStore/QAChain）
├── config.py           # YAML + .env + 默认值（三层优先级）
├── models.py           # Pydantic 数据模型
├── ingestion/
│   ├── loader.py       # 自动识别文件类型 → LangChain 加载器（PDF/TXT/MD/DOCX/HTML）
│   ├── splitter.py     # 固定长度：RecursiveCharacterTextSplitter；语义：委托 semantic_chunker
│   └── semantic_chunker.py  # 句子分割 → 滑动窗口 → 嵌入 → 余弦相似度 → 断点分块
├── embedding/
│   └── provider.py     # 工厂：创建 OpenAIEmbeddings（关闭 tiktoken，chunk_size=10）
├── storage/
│   └── vector_store.py # ChromaDB 封装：增/删/查/列表/计数/MMR + get_all_contents
├── retrieval/
│   ├── reranker.py     # Cross-Encoder 重排序（可选依赖，需 uv sync --extra reranker）
│   └── retriever.py    # similarity | mmr | hybrid（向量 + BM25 → RRF 融合）
└── chat/
    └── qa_chain.py     # LCEL 链：检索 → 格式化上下文 → LLM → 解析回答。ask() / chat()
```

### 数据流

**导入：** `文件 → loader → [splitter → semantic_chunker] → embedder → ChromaDB`

**检索：** `问题 → [多查询扩展 → 3个查询变体 → 分别搜索] → 合并去重 → LLM → 回答`

多查询扩展在 `qa_chain.py` 的 `_expand_query()` 中实现：将用户问题交给 LLM 生成 3 个不同角度的查询变体，分别检索后合并去重，提升召回率。

## 配置体系

| 配置类 | YAML 段 | 环境变量前缀 | 关键行为 |
|---|---|---|---|
| `LLMConfig` | `llm` | `LLM_` | `.env` 中的 api_key/api_base 覆盖 YAML |
| `EmbeddingConfig` | `embedding` | `EMBEDDING_` | provider.py 中硬编码关闭 tiktoken |
| `VectorStoreConfig` | `vector_store` | `VECTOR_STORE_` | persist_directory 默认 `data/chroma_db` |
| `IngestionConfig` | `ingestion` | `INGESTION_` | chunk_mode: semantic/fixed |
| `RetrievalConfig` | `retrieval` | `RETRIEVAL_` | search_type: similarity/mmr/hybrid；reranker_enabled 开启 Cross-Encoder 重排序 |
| `ServerConfig` | `server` | `SERVER_` | host/port/reload |

新增配置字段的步骤：
1. 在 `config.py` 对应的 `*Config` 类中添加字段
2. 在 `config.yaml` 对应段中添加 YAML 键值
3. `from_yaml()` 自动合并 YAML + 环境变量

## 关键设计决策

- **Embedding 兼容性**：`tiktoken_enabled=False`、`check_embedding_ctx_length=False` — 非 OpenAI API（如阿里云 DashScope）不接受 token ID 格式，必须发送原始文本。`chunk_size=10` 适配 DashScope 每批最多 10 条的限制。
- **构建系统**：hatchling 配置 `[tool.hatch.build.targets.wheel] packages = ["src"]` — 这很关键，否则 editable install 会把 `src/` 放到 `sys.path` 而非项目根目录。
- **hatchling 的副作用**：修改 pyproject.toml 后 uv 会自动重建包。如果构建失败（例如缺少 README.md），`uv sync` 也会失败。此时先创建最小 README.md 文件。
- **语义分块**：句子 → `window_size` 的滑动窗口分组 → 每个窗口嵌入 → 相邻窗口余弦相似度 → 相似度低于 `threshold_percentile` 百分位数的位臵为断点。
- **混合检索**：`search_type: hybrid` 启用向量 + BM25 关键词检索，通过 RRF（倒数排序融合）合并结果。BM25 索引从 ChromaDB 内容懒加载构建。**注意事项**：API 模式下，导入/删除文档后 BM25 索引会过时，需调用 `retriever.rebuild_bm25_index()` 同步。
- **重排序 (Reranker)**：`reranker_enabled: true` 启用。初检 `rerank_k` 条（默认 20）→ Cross-Encoder 逐对评分 → 取 top-k。可选的 `sentence-transformers` 依赖，需 `uv sync --extra reranker` 安装。模型首次使用时自动下载。
- **对话历史**：以 `conversation_id` 为 key 存储在内存 dict 中，重启后丢失。
- **`<source>` 参数**：使用 `rag list-docs` 输出的"来源路径"（例如 `test_docs/law.txt`）。

## 需求追踪

- `requirements.md` 追踪功能状态（`[x]` = 已完成，`[ ]` = 待实现）。
- 实现新功能或修复 Bug 后，同步更新 `requirements.md`：将已完成的标记为 `[x]`，新增功能点加入对应章节，已修复的问题登记在第十一节。
