# SmartShop-RAG

一个面向电商客服场景的检索增强问答项目。当前版本聚焦单一主线：基于本地知识库完成客服问答，提供 `Streamlit` 演示界面、`FastAPI` 服务接口、向量检索、RAG 摘要生成和最小检索链路可观测性。

## 项目定位

SmartShop-RAG 的目标不是做通用 Agent，而是做一个结构清晰、数据来源可信、便于继续扩展的电商客服 RAG 项目。当前仓库保留了最小可运行骨架，后续会围绕以下方向继续迭代：

- 混合检索：向量检索 + BM25
- 检索优化：query rewrite、rerank、去重与融合
- 证据展示：回答引用与检索 trace 展示
- 数据建设：使用公开、可解释的电商客服知识数据集替换旧 demo 数据

## 当前能力

- 电商客服问答主链路
- 本地知识库构建与向量检索
- RAG 摘要生成与最终回答生成
- `GET /health` 健康检查
- `POST /chat` 单一问答接口
- `Streamlit` 本地聊天演示页面
- 会话级消息存储与摘要
- 检索阶段状态事件和基础 trace

## 当前不包含

当前仓库已经明确移除以下主线能力：

- 报告生成
- 用户长期记忆 / 趋势记忆
- 自定义离线 evaluation 主线
- 多工具 Agent 工作流

这个项目当前只服务于一个目标：把电商客服 RAG 主链路做扎实。

## 技术栈

- Python
- Streamlit
- FastAPI
- LangChain
- Chroma
- DashScope
- PyYAML
- Requests

## 目录结构

```text
.
├─config/                         # 模型、向量库、提示词、会话存储配置
├─data/
│  ├─knowledge_base/              # 电商客服知识库原始文件
│  └─sessions/                    # 本地会话存储
├─docs/                           # 项目文档
├─prompts/                        # 主回答与 RAG 摘要提示词
├─src/
│  └─smartshop_rag/
│     ├─agent/                    # RAG 问答编排
│     ├─api/                      # FastAPI 接口
│     ├─model/                    # 模型工厂
│     ├─rag/                      # 检索与向量库
│     ├─services/                 # 会话、依赖、状态事件
│     ├─ui/                       # Streamlit UI 组件
│     ├─utils/                    # 通用工具
│     └─web/                      # Streamlit 启动与 bootstrap
├─tests/                          # 最小测试集
├─environment.yml                 # conda 环境配置
├─requirements.txt                # pip 依赖清单
├─pyproject.toml
└─README.md
```

## 系统流程

当前主链路比较简单：

1. 用户输入问题
2. 系统从本地知识库检索相关资料
3. RAG 模块先对检索结果做摘要整理
4. 主模型结合用户问题、会话上下文和检索证据生成最终回答
5. 同时记录基础状态事件和检索 trace

这是一条标准的 RAG 问答链路，不包含复杂 Agent 工具规划。

## 模型配置

模型角色在 `config/rag.yml` 中配置：

- `primary_chat`：最终回答生成
- `rag_chat`：检索摘要生成
- `embedding`：向量检索 embedding

## 环境准备

### 方式一：使用 conda

```powershell
conda env create -f environment.yml
conda activate smartshop-rag
```

### 方式二：使用 pip

```powershell
pip install -r requirements.txt
pip install python-dotenv==1.1.1
```

## 环境变量

至少需要配置：

- `DASHSCOPE_API_KEY`

可以在项目根目录创建 `.env`：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
```

## 数据准备

当前项目不再内置旧的扫地机器人 demo 数据。你需要自行准备新的电商客服知识文件，并放到：

```text
data/knowledge_base/
```

推荐放入的内容包括：

- 平台帮助中心规则
- 商家 FAQ
- 商品详情说明
- 售后 / 退换货 / 配送规则
- 公开商品问答或清洗后的客服知识文本

当前默认支持的文件类型由 `config/chroma.yml` 控制，现为：

- `txt`
- `pdf`

## 构建向量库

放入知识文件后，执行：

```powershell
python src/smartshop_rag/rag/ingest.py
```

成功后会在本地生成向量库目录 `chroma_db/`。

## 启动方式

### 启动 Streamlit

```powershell
streamlit run src/smartshop_rag/web/app.py
```

### 启动 FastAPI

```powershell
uvicorn smartshop_rag.api.main:app --app-dir src --reload
```

## API

当前仅保留两个接口：

- `GET /health`
- `POST /chat`

详细说明见 `docs/api_usage_guide.md`。

### 聊天接口示例

```powershell
Invoke-RestMethod -Method POST `
  -Uri "http://127.0.0.1:8000/chat" `
  -ContentType "application/json" `
  -Body '{"user_id":"demo_user","message":"这款商品支持七天无理由吗？"}'
```

## 可观测性

当前项目不再保留旧的 Agent 级复杂 trace，但仍保留最小可观测性：

- 状态事件：检索中、回答生成中
- 检索 trace：当前 query、召回文档数量
- 会话摘要：最近几轮用户与助手对话摘要

这部分是为了后续继续做混合检索和检索效果对比时，保留最基本的调试能力。

## 当前限制

- 当前检索主链路仍以向量检索为主
- 尚未接入 BM25、混合检索、rerank 和引用展示
- 当前知识库质量完全取决于你准备的数据集
- 当前项目只是 RAG 骨架，真正的亮点将来自后续的数据建设和检索优化

## 后续规划

下一阶段优先级：

1. 收集公开、可信的电商客服知识数据
2. 完成数据清洗和知识库重建
3. 引入 BM25 和混合检索
4. 增加 rerank 和引用展示
5. 再考虑是否补充社区认可的 RAG 评测方式
