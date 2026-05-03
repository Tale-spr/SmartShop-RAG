# SmartShop-RAG

SmartShop-RAG 是一个面向电商客服场景的垂直 RAG 问答系统，示例业务聚焦美的空气炸锅品类。系统围绕真实客服链路设计，覆盖商品参数、功能差异、使用说明、清洁保养、常见故障、售后政策和多轮型号确认等问题。

项目提供 Streamlit 对话界面、FastAPI 服务、本地知识库构建、混合检索、Agentic Workflow 编排、检索 Trace 和 Ragas 离线评测能力。回答策略以“有证据再回答、型号不明确先澄清、证据不足不编造”为核心。

## 项目亮点

- **垂直电商客服场景**：围绕商品详情、说明书、售后规则和用户真实问法组织知识库。
- **知识库建立**：Markdown 语义切块结果持久化为 `data/index/chunks.jsonl`，BM25 与 Chroma 向量库共享同一份切块结果、编号和元数据。
- **Markdown Section 合并切块**：按标题解析语义 section，将相邻短 section 合并为更完整的检索单元，避免过碎上下文导致“资料不足”。
- **混合检索链路**：向量检索负责语义召回，BM25 负责型号、参数、规则词命中，RRF 融合后进入重排。
- **检索后证据整理**：命中知识块后按证据类型调序、补齐问题中的型号和属性信息，并拼接必要的相邻上下文。
- **客服风险控制**：对“这款”“这个型号”等易错问题进行型号确认，无法确认时优先澄清。
- **可观测与可评测**：每次回答保留检索 trace、命中文档、状态事件；Ragas 评测脚本覆盖数据集构建、评分和报告生成。

## 数据规模

项目使用小规模、垂直领域知识库，重点展示完整 RAG 工程闭环，而不是海量数据堆叠。

| 口径 | 规模 |
|---|---:|
| 覆盖品类 | 美的空气炸锅 |
| 覆盖型号 | 10 个 |
| 入库清洗文档 | 44 份 Markdown |
| 原始素材 | 137 个文件，约 21.7 MB，含 42 张说明书截图 |
| 统一知识块 | 76 个 |
| 平均知识块长度 | 约 235 字 |
| 主查询集 | 86 条 |
| Ragas 标注评测集 | 35 条 |

当前知识库覆盖的文档类型包括：

- `detail`：商品定位、核心卖点、适用场景、典型问法
- `specs`：容量、功率、操作方式、尺寸、重量等规格参数
- `manual`：首次使用、清洁保养、常见问题和故障处理
- `service`：发货、签收、退换货、发票、保修
- `shared/policies`：跨型号通用政策知识

## 系统架构

```text
用户问题
  │
  ├─ Intent Router
  │   ├─ smalltalk / capability / non-domain
  │   └─ product / usage / fault / policy QA
  │
  ├─ Query Rewrite
  │
  ├─ Hybrid Retrieval
  │   ├─ Chroma vector search
  │   ├─ BM25 keyword search
  │   └─ Weighted RRF fusion
  │
  ├─ Rerank
  │
  ├─ Context Postprocess
  │   ├─ 按证据类型调序
  │   ├─ 补齐型号和属性证据
  │   └─ 扩展相邻上下文
  │
  ├─ Model Confirmation
  │
  └─ Grounded Answer
```

系统不是让模型自由决定是否检索，而是用固定工作流约束客服问答路径：先识别问题类型，再改写查询、检索证据、确认型号，最后基于证据生成回答。

## 知识库构建

知识库构建入口：

```powershell
$env:PYTHONPATH='src'
python src/smartshop_rag/rag/ingest.py
```

强制清空旧索引并重建：

```powershell
$env:PYTHONPATH='src'
python src/smartshop_rag/rag/ingest.py --reset
```

构建产物：

- `data/index/chunks.jsonl`：统一知识块清单
- `chroma_db/`：Chroma 本地向量库
- `md5.text`：兼容保留的旧入库记录文件

`chunks.jsonl` 的首行为清单元信息，记录 schema version、chunk strategy、源文件 MD5 和 fingerprint；后续每行是一个知识块，包含 `chunk_id`、`page_content`、`metadata`、`source_path`、`source_md5`、`chunk_index`。

## 切块策略

当前切块策略为 `markdown_semantic_merged`：

```yaml
chunk_strategy_version: 3
chunk_size: 650
chunk_overlap: 80
```

Markdown 文档先按标题解析为 section，再按语义关系合并短 section：

- `detail.md` 中的核心定位、核心卖点、适用场景可合并为一个更完整的商品证据块。
- `典型问法映射` 单独保留，避免把用户问法列表混入主要证据块。
- `manual.md` 中快速入门与清洁保养可合并，相邻 FAQ 可合并。
- `specs.md` 通常保持参数块完整。
- 单个 section 超过 `chunk_size` 时使用 `RecursiveCharacterTextSplitter` 递归切分，并保留 overlap。

知识块 metadata 保留标题路径和合并范围：

- `heading_path`
- `heading_path_list`
- `heading_level`
- `section_index_start`
- `section_index_end`
- `merged_section_count`
- `chunk_in_section_index`

## 检索与回答

默认检索配置位于 `config/rag.yml`。核心模式包括：

- `weighted_rrf`：向量检索与 BM25 的加权 RRF 融合。
- `weighted_rrf_v2`：针对明确型号、弱特征问题和泛化问题使用不同权重。
- `weighted_rrf_v2_rerank`：在融合结果上调用重排模型选择最终上下文。

检索后处理包含三层，目标是把“检索命中的片段”整理成“足够回答问题的证据上下文”：

- **相邻上下文扩展**：命中某个知识块后，按同一来源文件拼接前后相邻内容，避免只给模型一小段孤立片段。
- **证据类型调序**：把规格参数、核心卖点、说明书、售后政策等可直接支撑回答的内容排得更靠前，把“典型问法映射”这类问法提示排得更靠后。
- **型号与属性补全**：从问题中识别完整型号、短型号别名和容量、功率、清洗、故障等属性词，发现最终上下文缺少对应证据时补入相关知识块。

最终回答会结合用户问题、会话摘要、最近对话、检索证据和型号确认状态，输出正常回答或保守澄清。

## 评测结果

Ragas 主评测集包含 35 条标注样本，覆盖商品参数、功能差异、多型号对比、使用入门、故障处理、清洁保养、售后规则和适用场景。

当前 `weighted_rrf_v2_rerank_merged_chunks` 结果：

| 指标 | 分数 |
|---|---:|
| context_precision | 0.8127 |
| faithfulness | 0.8356 |
| answer_relevancy | 0.6108 |

类别表现：

- 商品参数、售后规则、适用场景类检索稳定性较高。
- 多型号对比类在合并 chunk 后获得更完整证据。
- 回答聚焦性仍是主要挑战，尤其是功能差异和复杂对比问题。

Ragas 官方不提供统一合格线，本项目按垂直客服场景使用固定测试集评估，并结合人工验收判断最终可用性。

## 快速开始

### 1. 创建环境

使用 conda：

```powershell
conda env create -f environment.yml
conda activate smartshop-rag
```

或使用 pip：

```powershell
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env`：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
```

### 3. 构建知识库

```powershell
$env:PYTHONPATH='src'
python src/smartshop_rag/rag/ingest.py --reset
```

### 4. 启动 Streamlit

```powershell
streamlit run src/smartshop_rag/web/app.py
```

### 5. 启动 FastAPI

```powershell
uvicorn smartshop_rag.api.main:app --app-dir src --reload
```

### 6. 运行测试

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests
```

## Ragas 评测

生成评测数据集：

```powershell
$env:PYTHONPATH='src'
python src/smartshop_rag/eval/build_ragas_dataset.py `
  --query-set data/query_sets/air_fryer_midea_query_set_main_v3.jsonl `
  --annotations data/eval/ragas/annotations/main_v3_reference_answers_v1.jsonl `
  --mode weighted_rrf_v2_rerank `
  --output data/eval/ragas/datasets/main_v3_ragas_dataset_weighted_rrf_v2_rerank_merged_chunks.jsonl
```

运行 Ragas：

```powershell
python src/smartshop_rag/eval/run_ragas_eval.py `
  --dataset-path data/eval/ragas/datasets/main_v3_ragas_dataset_weighted_rrf_v2_rerank_merged_chunks.jsonl
```

生成报告：

```powershell
python src/smartshop_rag/eval/analyze_ragas_results.py `
  --results-jsonl data/eval/ragas/results/main_v3_ragas_scores_weighted_rrf_v2_rerank_merged_chunks.jsonl `
  --output-report data/eval/ragas/reports/main_v3_ragas_analysis_weighted_rrf_v2_rerank_merged_chunks_2026_05_03.md
```

## API

核心接口：

- `GET /health`
- `POST /chat`

`POST /chat` 返回：

- `user_id`
- `session_id`
- `answer`
- `status_events`
- `session_summary`

详细字段见：[API 使用说明](docs/api_usage_guide.md)

## 项目结构

```text
.
├─config/                    # 模型、检索、Chroma、提示词配置
├─data/
│  ├─knowledge_base/         # 商品知识、说明书、规则知识
│  ├─query_sets/             # 主查询集与专项查询集
│  ├─index/                  # 本地知识块清单生成物
│  └─eval/ragas/             # Ragas 标注、数据集、结果和报告
├─docs/                      # API、数据组织、评测说明
├─prompts/                   # 路由、改写、重排、回答提示词
├─src/smartshop_rag/
│  ├─agent/                  # LangGraph 工作流与运行时状态
│  ├─api/                    # FastAPI 服务
│  ├─eval/                   # 离线评测脚本
│  ├─model/                  # 模型工厂与适配层
│  ├─rag/                    # 检索、切块、入库、向量库、BM25
│  ├─services/               # 会话、依赖、状态事件服务
│  ├─ui/                     # Streamlit UI 组件
│  └─web/                    # Web 应用装配
├─tests/                     # 单元测试与回归测试
└─README.md
```

## 技术栈

- Python 3.13
- LangChain / LangGraph
- Chroma
- BM25
- FastAPI
- Streamlit
- DashScope / Qwen
- Ragas
- unittest

## 相关文档

- [API 使用说明](docs/api_usage_guide.md)
- [数据组织说明](docs/数据组织说明.md)
- [项目评测说明](docs/项目评测说明.md)
- [Ragas 评测指南](docs/ragas_eval_guide.md)
