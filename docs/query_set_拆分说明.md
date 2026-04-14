# Query Set 拆分说明

## 目标
将 `air_fryer_midea_query_set_v1.jsonl` 拆分为两套用途明确的实验输入：

- `air_fryer_midea_query_set_main_v2.jsonl`
  - 用于评估混合检索、重排和正常电商客服问答能力
- `air_fryer_midea_query_set_model_confirmation_v1.jsonl`
  - 用于评估“型号未确认 / 已确认 / 冲突”这三类场景的处理能力

原始 `air_fryer_midea_query_set_v1.jsonl` 保留为历史基线，不做覆盖。

## 拆分规则
### main_v2
保留以下问题：
- query 已明确给出型号，如 `MF-KZ30E201`、`5004`、`6054`、`7001`
- shared policy / 售后规则类问题
- 多型号对比或全产品线对比类问题，只要 query 本身不依赖“先确认用户手上是哪一款”
- 知识外问题，只要核心测试点是检索与回答边界，而不是型号确认

### model_confirmation_v1
纳入以下问题：
- 用户没有给明确型号，只提供弱特征或模糊说法，如 `3L 这款`、`5L 这个型号`、`带可视窗那款`、`7L 这款`
- 明确给出型号后，系统应直接回答、不再重复核验的样本
- 额外补充的冲突样本，用于测试“用户说的型号”和页面特征/证据不一致时是否会回到澄清模式

## 当前文件说明
### `data/query_sets/air_fryer_midea_query_set_main_v2.jsonl`
- 来源：`v1` 中剔除型号确认专项问题后得到
- 样本数：45
- 用途：
  - 比较 `vector / bm25 / hybrid / hybrid_rerank`
  - 分析排序质量、证据命中和回答覆盖

### `data/query_sets/air_fryer_midea_query_set_model_confirmation_v1.jsonl`
- 来源：
  - `v1` 中抽取 15 条型号确认相关样本
  - 新增 4 条冲突样本
- 样本数：19
- 用途：
  - 分析是否误认型号
  - 分析确认后是否仍重复要求核验型号
  - 分析冲突时是否回到澄清模式

## 推荐运行方式
### 主能力实验
```powershell
python src/smartshop_rag/rag/experiment.py --query-set data/query_sets/air_fryer_midea_query_set_main_v2.jsonl --output data/query_sets/air_fryer_midea_experiment_main_v2.jsonl
```

### 型号确认专项实验
```powershell
python src/smartshop_rag/rag/experiment.py --query-set data/query_sets/air_fryer_midea_query_set_model_confirmation_v1.jsonl --output data/query_sets/air_fryer_midea_experiment_model_confirmation_v1.jsonl
```

## 说明
- 两个新集合都沿用原 JSONL 最小字段，不改 schema。
- 同一条 query 不会同时出现在两个新集合中。
- `model_confirmation_v1` 中新增的 4 条 `afc_` 样本用于覆盖冲突场景，当前 `v1` 中没有足够等价样本。
