# SmartShop-RAG

SmartShop-RAG 是一个面向电商客服场景的智能问答项目，当前聚焦空气炸锅品类。项目围绕真实客服链路设计，结合本地知识库、混合检索、Agentic Workflow和多轮会话能力，提供可评测的问答服务。

项目提供：

- Streamlit 对话演示界面
- FastAPI 接口服务
- 本地知识库构建与向量库持久化
- 离线评测与实验脚本

## 项目背景

电商客服类问题通常具有两个难点：

- 问题看似简单，但答案强依赖商品参数、说明书、售后规则等外部知识
- 用户常常不会准确提供型号，直接回答容易把相似商品的信息误答到当前商品上

SmartShop-RAG 的目标是一个贴近真实业务的垂直客服系统：在有证据时基于知识库回答，在证据不足或型号不明确时保持保守，避免编造。

## 核心能力

- Agentic Workflow：使用 LangGraph 编排意图路由、Query Rewrite、检索决策和回答生成
- 本地知识库问答：基于商品资料、说明书、清洁保养信息和售后规则进行回答
- 混合检索：结合向量检索与 BM25 召回，再做融合与重排
- 受限补充检索：当首轮证据不足时，仅允许一次 Query Transform 后再次检索
- 型号确认风控：针对“这款容量多少”“这个型号能不能退”这类高风险问题，结合 query 与检索结果判断是否需要先澄清型号
- Smalltalk 分流：对“你好”“在吗”“谢谢”“我是谁”这类轻交互，使用轻量模型单次回复，不进入 RAG 主链路
- 状态事件与 Trace：保留检索 trace、状态事件和命中文档，便于调试、评测和前端展示

## 系统工作流

默认问答链路如下：

1. Intent Router 判断问题类型，区分 `smalltalk`、`capability_query`、`non_domain` 和领域内问答
2. 领域内问题进入 Query Rewrite，生成更适合检索的查询
3. 执行混合检索与重排，产出 `retrieved_docs` 与 `retrieval_trace`
4. 刷新型号确认状态，判断当前证据是否足够
5. 若证据不足，则执行一次受控的 Query Transform 后补充检索
6. Answer Node 根据证据充分性和型号确认状态输出：
   - 正常回答
   - 保守澄清

这个流程的重点不是“让模型自由调用工具”，而是在客服场景下优先保证回答的稳定性、可解释性和业务风险控制。

## 技术方案

### 1. 知识库与数据组织

当前知识库围绕空气炸锅品类构建，数据来源包括：

- 商品详情页卖点与参数
- 使用说明与清洁保养资料
- 售后、退换货、发票、配送等规则信息

检索主数据位于 `data/knowledge_base/cleaned/`，文档按品牌、型号、文档类型组织，并在入库时写入元数据，例如：

- `brand`
- `model`
- `doc_type`
- `source_path`
- `chunk_id`

### 2. 检索与重排

项目使用基于 RRF 的混合检索模式 `weighted_rrf`：

- 向量检索负责召回语义相近内容
- BM25 负责补充关键词、型号、规则词命中
- 两路结果通过 RRF 融合统一排序
- 项目内同时保留带重排和带业务权重修正的扩展模式，用于实验和进一步优化

这种设计更适合电商客服场景，因为很多问题既依赖语义理解，也依赖型号、容量、功率、保修等明确关键词；使用 RRF 融合后，可以更稳定地兼顾两类召回信息。

### 3. 回答生成与风控

最终回答结合以下信息统一生成：

- 用户问题
- 会话摘要
- 最近原始对话（6条）
- 检索总结
- 命中文档证据
- 型号确认状态

其中型号确认是一个重点风控点。当用户没有明确提供型号，且检索结果涉及多个候选型号时，系统优先保守澄清，而不是直接给出具体参数或结论。

### 4. 模型分层

项目对不同节点使用不同模型角色：

- `primary_chat`：最终回答生成
- `smalltalk_chat`：smalltalk 轻量回复
- `rag_chat`：检索摘要相关任务
- `rewrite_chat`：Query Rewrite / Query Transform
- `rerank_chat`：候选重排
- `eval_chat`：离线评测
- `embedding`：向量化模型

目前 smalltalk 使用轻量模型单独处理，其余非最终回答节点默认关闭 thinking，以降低整体响应延迟。

## 技术栈

- Python
- LangChain
- LangGraph
- FastAPI
- Streamlit
- Chroma
- BM25
- Qwen / DashScope
- Ragas
- unittest

## 快速开始

### 1. 安装依赖

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

### 3. 构建本地向量库

```powershell
$env:PYTHONPATH=(Resolve-Path 'src')
python src/smartshop_rag/rag/ingest.py
```

### 4. 启动 Web 演示

```powershell
streamlit run src/smartshop_rag/web/app.py
```

### 5. 启动 API 服务

```powershell
uvicorn smartshop_rag.api.main:app --app-dir src --reload
```

### 6. 运行测试

```powershell
$env:PYTHONPATH=(Resolve-Path 'src')
python -m unittest discover -s tests
```

## API 概览

项目当前提供两个核心接口：

- `GET /health`
- `POST /chat`

`POST /chat` 用于发起对话请求，返回：

- `user_id`
- `session_id`
- `answer`
- `status_events`
- `session_summary`

详细字段说明见：[API 使用说明](docs/api_usage_guide.md)

## 项目结构

```text
.
├─config/                    # 模型、检索和提示词配置
├─data/
│  ├─knowledge_base/         # 商品知识、说明书、规则知识
│  ├─query_sets/             # 主测试集与专项测试集
│  └─eval/ragas/             # Ragas 标注与评测结果
├─docs/                      # API、数据组织、评测等说明文档
├─prompts/                   # 系统提示词、检索提示词、路由提示词
├─src/smartshop_rag/
│  ├─agent/                  # Agentic Workflow 与运行时状态
│  ├─api/                    # FastAPI 服务
│  ├─eval/                   # 离线评测与分析脚本
│  ├─model/                  # 模型工厂与适配层
│  ├─rag/                    # 检索、向量库、数据入库
│  ├─services/               # 会话、依赖、状态事件等服务
│  ├─ui/                     # Streamlit 页面组件
│  └─web/                    # Web 启动与装配
├─tests/                     # 核心测试
└─README.md
```

## 项目特点

- 面向真实业务场景，而不是通用聊天 Demo
- 同时覆盖商品参数、使用说明、清洁保养、售后规则和型号确认
- 保留检索 trace 与状态事件，便于调试和前端展示处理过程
- 兼顾问答效果、风险控制和响应延迟
- 具备本地演示、API 服务和离线评测三种使用方式

## 相关文档

- [API 使用说明](docs/api_usage_guide.md)
- [数据组织说明](docs/数据组织说明.md)
- [项目评测说明](docs/项目评测说明.md)
- [Ragas 评测指南](docs/ragas_eval_guide.md)