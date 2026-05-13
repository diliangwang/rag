# RAG 系统需求文档

> 基于 LangChain + ChromaDB 的通用文档问答系统
> 技术栈：Python 3.12 / uv / LangChain / ChromaDB / FastAPI / Typer

---

## 一、核心 RAG 流水线

每条实现记录格式：`[状态] 功能 - 说明`

- [x] **文档加载** — 支持从本地文件系统加载文档，自动识别文件格式并选用对应的加载器
- [x] **文本分割** — 使用 `RecursiveCharacterTextSplitter` 将长文档分割为指定大小的文本块
- [x] **向量嵌入** — 调用 Embedding API 将文本块转为向量表示
- [x] **向量存储** — 将向量及文本存入 ChromaDB（本地持久化）
- [x] **相似度检索** — 根据用户问题从向量库检索最相关的文档块
- [x] **LLM 生成** — 将检索结果作为上下文提交给 LLM，生成最终回答

---

## 二、文档导入 (Ingestion)

- [x] **单文件导入** — `rag ingest <file>` 导入单个文档
- [x] **目录批量导入** — `rag ingest <dir>` 递归导入目录下所有受支持的文档
- [x] **PDF 支持** — 使用 `PyPDFLoader`
- [x] **TXT 支持** — 使用 `TextLoader` (UTF-8)
- [x] **Markdown 支持** — 使用 `TextLoader` (UTF-8)
- [x] **DOCX 支持** — 使用 `Docx2txtLoader`
- [x] **HTML 支持** — 使用 `BSHTMLLoader`（依赖 `beautifulsoup4` + `lxml`）
- [x] **分块参数可配置** — `chunk_size` / `chunk_overlap` 可通过 config.yaml 或 CLI 参数覆盖
- [x] **导入进度反馈** — 显示导入的段落数和分块数，每个文件标记成功/失败
- [x] **语义分块** — 基于文本语义相似度的智能分块：将文本拆分为句子后分组嵌入，相邻组间余弦相似度低于阈值的位臵即为断点
  - 支持 `window_size`（滑动窗口大小）、`threshold_percentile`（断点阈值百分位）可配置
  - 低于 `min_chunk_chars` 自动合并，超过 `max_chunk_chars` 自动分割
  - 嵌入失败时自动回退到段落级分割

### 待优化
- [ ] **更多格式** — CSV / JSON / EPUB / 图片OCR
- [ ] **URL 直接导入** — 通过 URL 抓取网页内容并导入

---

## 三、向量存储 (Vector Store)

- [x] **ChromaDB 持久化** — 向量数据持久化到 `data/chroma_db/`
- [x] **文档管理** — `rag list-docs` 列出所有已导入文档（含每个源文件的块数）
- [x] **文档删除** — `rag delete <source>` 按源路径删除文档，返回实际删除块数
- [x] **文档块查看** — `rag chunks <source>` 查看指定文档的所有分块内容（支持 `-a` 显示完整内容）
- [x] **向量库统计** — 显示知识库总文档块数
- [x] **ChromaDB where 过滤** — 删除和查询均使用 `where` 条件直接过滤，避免全量拉取

### 待优化
- [ ] **多种向量库支持** — 支持 FAISS / Qdrant / Milvus 等后端切换
- [ ] **向量库备份/恢复** — 导出/导入向量数据
- [ ] **集合管理** — 支持多个命名集合（多知识库隔离）

---

## 四、检索 (Retrieval)

- [x] **相似度搜索** — 基于余弦相似度的向量检索
- [x] **MMR 搜索** — Maximal Marginal Relevance 搜索，兼顾相关性与多样性
- [x] **检索数量可配置** — `k` 值通过 config.yaml 配置
- [x] **相似度阈值** — `score_threshold` 过滤低质量结果
- [x] **多查询扩展** — 用 LLM 将用户问题改写为 3 个不同角度的查询，分别检索后合并去重
- [x] **来源引用** — 回答中附带来源文件名和相关度分数
- [x] **混合检索 (Hybrid)** — 向量检索（语义） + BM25（关键词） → RRF 倒数排序融合
  - 配置 `search_type: hybrid` 启用
  - 使用 `rank_bm25` 构建关键词索引，支持懒加载和手动重建
  - RRF 常数 `rrf_k` 和 BM25 独立 k 值 `hybrid_k` 可配置
- [x] **重排序 (Reranker)** — Cross-Encoder 对初检结果重评分，提升排序精度
  - 配置 `reranker_enabled: true` 启用
  - 使用 `cross-encoder/ms-marco-MiniLM-L-6-v2` 模型，逐对计算（query, doc）相关性分数
  - 初检 `rerank_k` 条（默认 20），重排序后取 top-k
  - 依赖 `sentence-transformers`，需 `uv sync --extra reranker` 安装

### 待优化
- [ ] **HyDE** — 假设文档嵌入，先让 LLM 生成假设答案再检索
- [ ] **检索调试输出** — 展示每个查询扩展生成了哪些检索词

---

## 五、问答 / 对话 (QA & Chat)

- [x] **单轮问答** — `rag ask <question>` 基于知识库回答单个问题
- [x] **多轮对话** — `rag chat` 交互式连续对话，保留上下文历史
- [x] **来源展示** — 回答后显示引用来源文件名和相似度
- [x] **对话历史管理** — 支持 `clear` 清空对话历史，`exit` 退出
- [x] **严格基于上下文** — 提示词要求 LLM 只基于提供的资料回答，禁止编造
- [x] **引用标注** — 提示词引导 LLM 在回答中标注来源文件

### 待优化
- [ ] **流式输出** — 支持 Stream 模式，逐字显示 LLM 回答
- [ ] **对话历史持久化** — 重启后恢复历史对话
- [ ] **Prompt 模板可自定义** — 允许用户自定义 QA 提示词模板
- [ ] **答案置信度** — 基于检索分数输出置信度标识
- [ ] **追问建议** — LLM 根据上下文自动生成建议的追问问题

---

## 六、CLI 命令行

- [x] **`rag ingest`** — 导入文档（文件或目录）
- [x] **`rag list-docs`** — 列出所有文档及块数
- [x] **`rag delete`** — 删除指定文档（未匹配时提示正确路径）
- [x] **`rag chunks`** — 查看文档的每个块内容（默认截取 300 字，`-a` 显示全文）
- [x] **`rag ask`** — 单轮问答
- [x] **`rag chat`** — 交互式多轮对话
- [x] **`rag retrieve`** — 调试命令，展示检索结果及多查询扩展
- [x] **`rag serve`** — 启动 API 服务
- [x] **全局错误处理** — API 密钥错误 / 文件未找到等友好提示，无堆栈跟踪
- [x] **Rich 格式化输出** — 表格、面板、彩色状态指示

### 待优化
- [ ] **Shell 自动补全** — 支持 `--install-completion`
- [ ] **`rag stats`** — 知识库统计信息（文档数、块数、向量库大小）
- [ ] **`rag export`** — 导出问答记录

---

## 七、API 服务

- [x] **`GET /health`** — 健康检查
- [x] **`POST /ingest`** — 上传文档并导入（验证文件格式）
- [x] **`GET /documents`** — 列出文档
- [x] **`DELETE /documents/{source}`** — 删除文档（404 处理）
- [x] **`POST /ask`** — 单轮问答
- [x] **`POST /chat`** — 多轮对话（支持 `conversation_id`）
- [x] **OpenAI 异常处理** — 统一返回 503 + 错误信息
- [x] **自动 API 文档** — FastAPI 内嵌 Swagger UI (`/docs`)

### 待优化
- [ ] **API 认证** — 添加 API Key 认证机制
- [ ] **速率限制** — 请求频率控制
- [ ] **CORS 配置** — 可配置跨域访问
- [ ] **批量文档上传** — 支持 zip 压缩包批量导入
- [ ] **异步任务队列** — 大文档导入后台处理，避免超时
- [ ] **OpenAPI 文档完善** — 添加请求/响应示例

---

## 八、配置管理

- [x] **YAML 配置文件** — `config.yaml` 集中管理所有配置
- [x] **环境变量覆盖** — `.env` 文件中的 `LLM_*` / `EMBEDDING_*` 变量优先级高于 YAML
- [x] **Pydantic 配置模型** — 类型校验 + IDE 自动补全
- [x] **多组件配置** — LLM / Embedding / 向量库 / 文档导入 / 检索 / 服务器 各模块独立配置
- [x] **`.env.example` 模板** — 提供配置模板，含 OpenAI 和 SiliconFlow 两种示例
- [x] **`load_dotenv`** — 配置加载时自动读取 `.env` 文件

### 待优化
- [ ] **配置热重载** — 运行时修改配置自动生效
- [ ] **多环境配置** — 支持 `config.{env}.yaml`（dev / test / prod）
- [ ] **配置验证** — 启动时校验 API 连通性

---

## 九、LLM 多提供商支持

- [x] **OpenAI 兼容接口** — 通过 `ChatOpenAI` 统一调用
- [x] **灵活切换** — 修改 `api_base` + `model` 即可切换提供商
- [x] **已实测兼容** — OpenAI / 阿里云 DashScope（通义千问）
- [x] **Embedding 适配** — `chunk_size=10` 适配 DashScope 批量限制
- [x] **Embedding 文本模式** — 关闭 tiktoken 分词，发送原始文本以避免非 OpenAI API 不兼容

### 待优化
- [ ] **更多提供商预配置** — 内置 SiliconFlow / Claude / 智谱 / DeepSeek 等预设
- [ ] **多模型路由** — 不同模型用于不同任务（检索 vs 生成 vs 重排序）
- [ ] **费用统计** — 记录每次 API 调用的 Token 消耗和费用

---

## 十、工程化 & 项目配置

- [x] **uv 依赖管理** — `pyproject.toml` 声明式依赖，`uv.lock` 锁定版本
- [x] **src 布局** — 标准 Python src 布局
- [x] **hatchling 构建** — `[build-system]` + `[tool.hatch.build]` 配置
- [x] **console_scripts 入口** — `rag` 命令全局可用
- [x] **`.gitignore`** — Python / 虚拟环境 / 环境变量 / 向量数据 规则
- [x] **类型注解** — 全项目类型标注

### 待优化
- [ ] **单元测试** — pytest 测试覆盖核心模块
- [ ] **CI/CD** — GitHub Actions 自动化测试
- [ ] **Docker 容器化** — Dockerfile + docker-compose
- [ ] **日志系统** — 结构化日志（loguru / structlog）
- [ ] **Makefile** — 常用命令快捷入口
- [ ] **CLAUDE.md** — 项目开发规范文档

---

## 十一、已修复的问题

| 日期 | 问题 | 解决方案 |
|------|------|----------|
| 2026-05-12 | Embedding 配置不读取 `.env` | `from_yaml()` 开头加入 `load_dotenv()` |
| 2026-05-12 | Embedding API 发送 token ID 而非文本 | 设置 `tiktoken_enabled=False` + `check_embedding_ctx_length=False` |
| 2026-05-12 | DashScope 批量嵌入超过 10 条限制 | 设置 `chunk_size=10` |
| 2026-05-12 | `rag delete` 无匹配时也显示成功 | 返回实际删除块数，未匹配时提示正确路径 |
| 2026-05-12 | 检索召回率不足导致漏回答 | 增加 `k=8` + 实现多查询扩展检索 |
| 2026-05-13 | `_split_sentences` 中 `re.split` 消耗句尾标点 | 改用捕获组 `([。！？：.!?\n]+)` 保留标点，按 `text+delimiter` 配对重建句子 |

---

## 十二、未来扩展规划

- [ ] **Web 前端** — 基于 Next.js / Streamlit 的图形界面
- [ ] **多用户支持** — 用户隔离的知识库
- [ ] **文档版本管理** — 追踪文档变更历史，增量更新向量库
- [ ] **RAG 评估** — 召回率 / 准确率 / 回答质量评估指标
- [ ] **Agent 集成** — 支持 LangGraph Agent，结合工具调用
- [ ] **多模态 RAG** — 支持图片理解（Vision LLM）
- [ ] **知识图谱融合** — GraphRAG，结合知识图谱增强检索
- [ ] **增量学习** — 持续学习用户反馈，优化检索排序

---

> **维护说明：**
> - `[x]` 表示已完成，`[ ]` 表示待实现
> - 每新增一个功能点，在此文档追加条目并标记状态
> - 修复 Bug 在第 11 节登记
