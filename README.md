# SmartShop-RAG

一个面向电商客服场景的检索增强问答项目，当前聚焦某品牌部分型号空气炸锅商品知识问答。项目提供本地知识库、混合检索与重排、Web 演示界面、API 服务，以及一套可复现的评测链路。

## 项目定位

SmartShop-RAG 是一个围绕“客服问答主链路”展开的 RAG 项目。它关注三个核心问题：

- 如何把公开商品信息、说明书和售后规则整理成可解释知识库
- 如何用混合检索和重排提升客服问答的证据命中率

## 核心能力

- 本地知识库构建与向量库持久化
- 向量检索、BM25、混合检索与加权融合实验
- Query rewrite、候选重排、检索 trace 记录
- 面向客服场景的最终回答生成
- Streamlit 聊天演示页面
- FastAPI 服务接口
- Ragas 离线评测与实验分析脚本

## 系统设计

### 数据组织

知识库当前围绕空气炸锅品类构建，数据来源包括：

- 商品详情页卖点与参数
- 说明书常见问题与清洁保养信息
- 售后、退换货、发票、配送等公共规则

知识以 `cleaned/` 文档为检索主数据源，按品牌、型号、文档类型组织，并在向量库中补充：

- `brand`
- `model`
- `doc_type`
- `source_path`
- `chunk_id`

### 检索链路

当前主链路采用：

1. Query rewrite
2. 双路召回：向量检索 + BM25
3. 候选融合
4. 轻量重排
5. 基于证据的回答生成

项目内保留了多种检索模式，用于实验对比和误差分析；对外展示的默认主链路聚焦在稳定可用的混合检索与重排方案上。

### 回答生成

回答阶段结合：

- 用户问题
- 检索到的证据文本
- 会话摘要
- 型号确认状态

目标是尽量给出基于当前知识库的客服式回答，并在型号不明确时保持保守。

## 项目亮点

- 知识来源公开且可解释，便于追踪回答依据
- 同时建设了主测试集、型号确认专项集和 Ragas 标注集
- 检索与回答链路可观测，便于分析命中、排序和 groundedness 问题
- 支持本地演示、API 服务和离线评测三种使用方式

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

### 3. 构建向量库

```powershell
$env:PYTHONPATH=(Resolve-Path 'src')
python src/smartshop_rag/rag/ingest.py
```

### 4. 启动演示或 API

启动 Streamlit：

```powershell
streamlit run src/smartshop_rag/web/app.py
```

启动 FastAPI：

```powershell
uvicorn smartshop_rag.api.main:app --app-dir src --reload
```

## API

当前对外接口保持精简：

- `GET /health`
- `POST /chat`

接口字段说明见：[API 使用说明](docs/api_usage_guide.md)

## 实验与评测

当前项目保留一套完整的离线评测方法，但 README 只展示最终口径下值得保留的结论：

- 主检索链路采用混合检索与重排，当前表现整体优于单一检索方式
- 检索层总体较稳，当前主要优化空间更多在回答贴题性与 groundedness
- 型号确认场景需要单独测试，不能与普通问答主测试集混在一起
- Ragas 用于补充验证上下文相关性、回答可信度和回答相关性，而不是替代业务专项测试

详细说明见：[项目评测说明](docs/项目评测说明.md)

## 目录结构

```text
.
├─config/                    # 模型、检索、向量库配置
├─data/
│  ├─knowledge_base/         # 商品知识与规则知识
│  ├─query_sets/             # 主测试集、专项测试集、Ragas 候选集
│  └─eval/ragas/             # 标注、数据集、评测结果
├─docs/                      # API、数据组织、评测说明等文档
├─prompts/                   # 回答生成与检索相关提示词
├─src/smartshop_rag/
│  ├─api/                    # FastAPI 服务
│  ├─web/                    # Streamlit 启动与装配
│  ├─rag/                    # 检索、向量库、实验脚本
│  ├─model/                  # 模型工厂与模型适配层
│  ├─services/               # 会话、依赖与状态事件
│  └─eval/                   # Ragas 数据集构建与结果分析
├─tests/                     # 核心测试
└─README.md
```

## 相关文档

- [API 使用说明](docs/api_usage_guide.md)
- [数据组织说明](docs/数据组织说明.md)
- [项目评测说明](docs/项目评测说明.md)
- [Ragas 评测指南](docs/ragas_eval_guide.md)

## 后续方向

- 继续扩充知识库中的型号覆盖与规则覆盖
- 优化混合检索融合策略和回答 groundedness
- 扩展更完整的客服专项测试与评测样本
- 在保持可解释性的前提下提升回答稳定性
