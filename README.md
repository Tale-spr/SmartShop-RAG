# SmartShop-RAG

一个面向电商客服场景的混合检索 RAG 项目骨架，当前保留 `Streamlit` 演示界面、`FastAPI` 服务接口、本地知识库构建、向量检索与基础检索链路埋点。

## 项目定位

- 单一主线：电商客服问答
- 技术核心：本地知识库 + RAG 摘要 + 最终回答生成
- 当前阶段：纯 RAG 骨架，后续将在此基础上补混合检索、BM25、重排与引用展示

## 当前能力

- `GET /health` 健康检查
- `POST /chat` 单一问答接口
- `Streamlit` 本地演示页面
- 会话级消息存储与摘要
- 向量库构建与检索
- 检索阶段状态事件与基础 trace

## 目录结构

```text
.
├─config/                         # 模型、向量库、提示词、会话存储配置
├─data/
│  ├─knowledge_base/              # 新知识库原始文件目录
│  └─sessions/                    # 本地会话文件存储
├─docs/                           # 重构说明与 API 文档
├─prompts/                        # 主回答与 RAG 摘要提示词
├─src/
│  └─smart_clean_agent/
│     ├─agent/                    # RAG 问答编排
│     ├─api/                      # FastAPI 接口
│     ├─model/                    # 模型工厂
│     ├─rag/                      # 检索与向量库
│     ├─services/                 # 会话、依赖、状态事件
│     ├─ui/                       # Streamlit UI 组件
│     ├─utils/                    # 通用工具
│     └─web/                      # Streamlit 启动与 bootstrap
├─tests/                          # 最小测试集
├─pyproject.toml
├─requirements.txt
└─README.md
```

## 模型配置

当前模型角色在 [rag.yml](/e:/Python/Agent项目/config/rag.yml) 中配置：

- `primary_chat`：最终回答生成
- `rag_chat`：检索摘要生成
- `embedding`：向量检索 embedding

## 本地运行

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 配置环境变量

至少需要：

- `DASHSCOPE_API_KEY`

### 3. 准备知识库文件

把电商客服知识文件放到：

```text
data/knowledge_base/
```

支持的知识文件类型由 [chroma.yml](/e:/Python/Agent项目/config/chroma.yml) 控制，当前为 `txt` 和 `pdf`。

### 4. 构建向量库

```powershell
python src/smart_clean_agent/rag/ingest.py
```

### 5. 启动 Streamlit

```powershell
streamlit run src/smart_clean_agent/web/app.py
```

### 6. 启动 FastAPI

```powershell
uvicorn smart_clean_agent.api.main:app --app-dir src --reload
```

## API

当前仅保留两个接口：

- `GET /health`
- `POST /chat`

详细说明见 [API 使用说明](/e:/Python/Agent项目/docs/api_usage_guide.md)。

### 聊天接口示例

```powershell
Invoke-RestMethod -Method POST `
  -Uri "http://127.0.0.1:8000/chat" `
  -ContentType "application/json" `
  -Body '{"user_id":"demo_user","message":"这款商品支持七天无理由吗？"}'
```

## 当前限制

- 当前检索主链路仍以向量检索为主，混合检索、BM25、重排与引用展示尚未接入
- 当前仓库已删除报告生成、用户长期记忆与自定义 evaluation 主线
- 当前知识库内容需要自行替换为新的电商客服数据集
