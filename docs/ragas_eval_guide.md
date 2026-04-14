# Ragas 评测接入指南

## 目标

这条评测链路只服务 `main_v2` 主测试集，用来回答四个问题：

- 混合检索和重排是否提升了上下文相关性
- 回答是否基本建立在检索到的上下文上
- 哪些类别问题最弱
- 当前更像是检索问题，还是回答 groundedness 问题

`model_confirmation_v1` 继续保留为独立专项测试集，不强行塞进 Ragas。

## 目录结构

- `src/smartshop_rag/eval/`
  - `build_ragas_dataset.py`
  - `run_ragas_eval.py`
  - `analyze_ragas_results.py`
- `data/eval/ragas/`
  - `annotations/`
  - `datasets/`
  - `results/`
  - `reports/`

## 环境准备

先安装 Ragas 依赖：

```powershell
pip install ragas==0.3.9 datasets==4.4.1
```

如果你发现 `ragas` 在当前 Python 3.13 环境中不稳定，优先切到 Python 3.11，而不是继续硬扛当前环境。

## 模型配置

当前项目已经新增 `eval_chat` 角色，默认仍走轻量模型：

```yaml
models:
  eval_chat: qwen-flash
```

Ragas 运行时会把：

- `eval_chat` 包装成 `LangchainLLMWrapper`
- 现有 DashScope embedding 包装成 `LangchainEmbeddingsWrapper`

## 第一步：准备 reference 标注

第一阶段不要一口气标完全部 `main_v2`。先用当前仓库里的最小标注集：

- `data/eval/ragas/annotations/main_v2_reference_answers_v1.jsonl`

格式示例：

```json
{"id": "af_002", "reference": "MF-KZ30E201 是 3L 小容量款，机身相对紧凑，更适合宿舍或小空间使用。"}
```

## 第二步：构建 Ragas 数据集

```powershell
$env:PYTHONPATH=(Resolve-Path 'src')
python src/smartshop_rag/eval/build_ragas_dataset.py --query-set data/query_sets/air_fryer_midea_query_set_main_v2.jsonl --annotations data/eval/ragas/annotations/main_v2_reference_answers_v1.jsonl --mode hybrid_rerank --output data/eval/ragas/datasets/main_v2_ragas_dataset_2026_04_14.jsonl
```

输出字段至少包括：

- `user_input`
- `retrieved_contexts`
- `response`
- `reference`

## 第三步：运行 Ragas

先跑不依赖 reference 的核心指标：

```powershell
$env:PYTHONPATH=(Resolve-Path 'src')
python src/smartshop_rag/eval/run_ragas_eval.py --dataset-path data/eval/ragas/datasets/main_v2_ragas_dataset_2026_04_14.jsonl
```

如果 reference 与 embedding wrapper 都正常，再补带 reference 的指标：

```powershell
$env:PYTHONPATH=(Resolve-Path 'src')
python src/smartshop_rag/eval/run_ragas_eval.py --dataset-path data/eval/ragas/datasets/main_v2_ragas_dataset_2026_04_14.jsonl --with-reference-metrics
```

当前第一阶段重点指标：

- `ContextPrecision`
- `Faithfulness`
- `AnswerRelevancy`

第二阶段再补：

- `ContextRecall`
- `AnswerCorrectness`

## 第四步：分析结果

```powershell
$env:PYTHONPATH=(Resolve-Path 'src')
python src/smartshop_rag/eval/analyze_ragas_results.py --results-jsonl data/eval/ragas/results/main_v2_ragas_scores_2026_04_14.jsonl
```

分析报告会写到：

- `data/eval/ragas/reports/`

## 如何读结果

### `ContextPrecision`
检索回来的上下文是否真的相关。这个分低，优先怀疑检索或重排。

### `Faithfulness`
回答是否建立在上下文上。这个分低，优先怀疑回答在补脑或约束不够。

### `AnswerRelevancy`
回答是否真正围绕用户问题。这个分低，优先怀疑 prompt 或回答结构跑偏。

### `ContextRecall`
关键证据是否被找回。这个指标更依赖 reference 标注质量。

### `AnswerCorrectness`
回答是否接近 reference。若它低但 `Faithfulness` 高，先检查 reference 是否写得太理想化，或知识库本身覆盖不够。

## 推荐执行顺序

1. 先跑 `build_ragas_dataset.py`
2. 先跑核心三指标
3. 再决定是否打开 reference 指标
4. 最后结合低分样本和 retrieval trace 做问题定位
