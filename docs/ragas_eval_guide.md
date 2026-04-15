# Ragas 评测指南

## 目标

Ragas 在本项目中的作用是补充评估检索上下文质量和回答可信度，而不是替代主测试集和专项测试集。

当前主要关注三项指标：

- `ContextPrecision`
- `Faithfulness`
- `AnswerRelevancy`

## 目录结构

```text
src/smartshop_rag/eval/
data/eval/ragas/
├─annotations/
├─datasets/
├─results/
└─reports/
```

## 环境准备

```powershell
pip install ragas==0.3.9 datasets==4.4.1
```

如当前 Python 版本与 `ragas` 兼容性不稳定，优先使用更稳定的环境再运行评测。

## 模型配置

Ragas 评测通过 `eval_chat` 角色调用评测模型，通过现有 embedding 模型提供向量表示。评测模型建议与实验链路保持一致，避免跨模型比较带来口径漂移。

## 第一步：准备标注

当前项目使用独立标注文件提供 reference，例如：

- `data/eval/ragas/annotations/main_v3_reference_answers_v1.jsonl`

示例：

```json
{"id": "afm_001", "reference": "示例参考答案"}
```

## 第二步：构建 Ragas 数据集

```powershell
$env:PYTHONPATH=(Resolve-Path 'src')
python src/smartshop_rag/eval/build_ragas_dataset.py --query-set data/query_sets/air_fryer_midea_query_set_main_v3.jsonl --annotations data/eval/ragas/annotations/main_v3_reference_answers_v1.jsonl --mode hybrid_rerank --output data/eval/ragas/datasets/main_v3_ragas_dataset_hybrid_rerank.jsonl
```

输出数据至少包含：

- `user_input`
- `retrieved_contexts`
- `response`
- `reference`

## 第三步：运行 Ragas

```powershell
$env:PYTHONPATH=(Resolve-Path 'src')
python src/smartshop_rag/eval/run_ragas_eval.py --dataset-path data/eval/ragas/datasets/main_v3_ragas_dataset_hybrid_rerank.jsonl
```

如果需要带 reference 的指标，再追加：

```powershell
$env:PYTHONPATH=(Resolve-Path 'src')
python src/smartshop_rag/eval/run_ragas_eval.py --dataset-path data/eval/ragas/datasets/main_v3_ragas_dataset_hybrid_rerank.jsonl --with-reference-metrics
```

## 第四步：分析结果

```powershell
$env:PYTHONPATH=(Resolve-Path 'src')
python src/smartshop_rag/eval/analyze_ragas_results.py --results-jsonl data/eval/ragas/results/main_v3_ragas_scores_hybrid_rerank.jsonl
```

## 如何解读结果

- `ContextPrecision` 低：优先怀疑检索与候选融合
- `Faithfulness` 低：优先怀疑回答对证据的依赖不够强
- `AnswerRelevancy` 低：优先怀疑回答结构不够贴题

Ragas 的作用是帮助定位问题，而不是给项目下单一结论。最终判断仍应结合主测试集、专项测试集和人工抽样。
