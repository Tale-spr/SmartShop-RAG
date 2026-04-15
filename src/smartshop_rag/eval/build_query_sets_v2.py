from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"E:\Python\SmartShop-RAG")
QUERY_DIR = ROOT / "data" / "query_sets"
ANNOTATION_DIR = ROOT / "data" / "eval" / "ragas" / "annotations"

entries: list[dict] = []
references: list[dict] = []
_counter = 1


def add_entry(*, query: str, category: str, target_models: list[str], difficulty: str,
              query_scope: str, model_clarity: str, usage_tags: list[str],
              reference: str | None = None) -> str:
    global _counter
    entry_id = f"aq_{_counter:03d}"
    _counter += 1
    entry = {
        "id": entry_id,
        "query": query,
        "category": category,
        "target_models": target_models,
        "difficulty": difficulty,
        "query_scope": query_scope,
        "model_clarity": model_clarity,
        "usage_tags": usage_tags,
    }
    entries.append(entry)
    if reference is not None:
        references.append({"id": entry_id, "reference": reference})
    return entry_id


models = [
    {
        "model": "MF-KZ26E101",
        "short": "26E101",
        "capacity": "2.6L",
        "power": "1300W",
        "param_query": "MF-KZ26E101 的容量和功率大概是多少？",
        "param_ref": "MF-KZ26E101 页面卖点显示为 2.6L 小容量，额定功率约 1300W。",
        "scenario_query": "2.6L 这款一个人用够吗？",
        "scenario_ref": None,
        "feature_query": "26E101 是机械旋钮款还是按键款？",
        "feature_ref": None,
        "usage_query": "MF-KZ26E101 首次使用前要做哪些准备？",
        "usage_ref": "首次使用 MF-KZ26E101 前，建议先清洗烤盘和炸桶等接触食材的部件，再放入食材，通过时间和温度旋钮开始烹饪。",
        "clean_query": "26E101 清洗时能不能直接放洗碗机？",
        "clean_ref": None,
        "fault_query": "26E101 插电后指示灯不亮一般先检查什么？",
        "fault_ref": "如果 MF-KZ26E101 插电后旋转定时器仍不亮，先检查炸篮插头是否插好、电源是否接通；若已出风但灯不亮，则可能是灯或线路问题。",
    },
    {
        "model": "MF-KZ30E201",
        "short": "30E201",
        "capacity": "3L",
        "power": "1325W",
        "param_query": "MF-KZ30E201 的容量和功率大概是多少？",
        "param_ref": None,
        "scenario_query": "3L 这款放宿舍用会不会太大？",
        "scenario_ref": "MF-KZ30E201 是 3L 小容量款，更适合宿舍、小厨房和小户型场景，一般不会太大。",
        "feature_query": "30E201 属于旋钮款还是按键款？",
        "feature_ref": "MF-KZ30E201 是机械旋钮款，不是按键或触控面板。",
        "usage_query": "第一次用 MF-KZ30E201 之前要先做什么？",
        "usage_ref": None,
        "clean_query": "30E201 清洗时能不能用钢丝球？",
        "clean_ref": None,
        "fault_query": "30E201 如果不工作，我应该先检查什么？",
        "fault_ref": "如果 MF-KZ30E201 不工作，先检查电源是否接通、定时器是否设定，以及炸篮是否完全安装到位。",
    },
    {
        "model": "MF-KZE4012",
        "short": "4012",
        "capacity": "4.2L",
        "power": "1400W",
        "param_query": "MF-KZE4012 的容量和功率大概是多少？",
        "param_ref": None,
        "scenario_query": "4.2L 这款更适合两个人还是三个人用？",
        "scenario_ref": "MF-KZE4012 属于 4.2L / 4L 容量段，通常更适合两人到三人的日常少量到中等份量烹饪。",
        "feature_query": "4012 这种基础款是机械旋钮还是触控面板？",
        "feature_ref": None,
        "usage_query": "MF-KZE4012 第一次使用前要不要先清洗炸桶？",
        "usage_ref": "首次使用 MF-KZE4012 前，建议先清洗炸桶和烤盘，再按旋钮设定时间开始使用。",
        "clean_query": "4012 日常清洁时能不能用强力去油清洁剂？",
        "clean_ref": None,
        "fault_query": "4012 工作时冒白烟一般先看哪里？",
        "fault_ref": None,
    },
    {
        "model": "MF-KZE459X9BD",
        "short": "459X9BD",
        "capacity": "4.5L",
        "power": "1350W",
        "param_query": "459X9BD 的容量和功率是多少？",
        "param_ref": "MF-KZE459X9BD 的容量约为 4.5L，额定功率约为 1350W。",
        "scenario_query": "4.5L 带可视窗这款更适合两个人还是三个人用？",
        "scenario_ref": None,
        "feature_query": "459X9BD 的可视窗实际有什么用？",
        "feature_ref": "MF-KZE459X9BD 的可视窗主要用于更直观地观察食材状态，减少频繁抽拉炸桶查看。",
        "usage_query": "MF-KZE459X9BD 首次使用前需要做哪些准备？",
        "usage_ref": None,
        "clean_query": "459X9BD 清洁的时候能不能用钢丝球？",
        "clean_ref": "MF-KZE459X9BD 清洁时不建议使用钢丝球、刀片等坚硬物品，以免刮花烤盘和炸桶。",
        "fault_query": "459X9BD 炸出来效果不理想一般先看哪里？",
        "fault_ref": None,
    },
    {
        "model": "MF-KZE5004",
        "short": "5004",
        "capacity": "5L",
        "power": "1375W",
        "param_query": "5004 这款容量和功率大概是多少？",
        "param_ref": "MF-KZE5004 页面卖点显示为 5L，额定功率约 1375W。",
        "scenario_query": "5L 这款日常家用够不够用？",
        "scenario_ref": None,
        "feature_query": "MF-KZE5004 是机械旋钮还是触控面板？",
        "feature_ref": None,
        "usage_query": "MF-KZE5004 首次使用前要不要先洗炸桶？",
        "usage_ref": "MF-KZE5004 首次使用前建议先清洗炸桶和烤盘。",
        "clean_query": "5004 这款平时怎么清洗比较合适？",
        "clean_ref": None,
        "fault_query": "5004 炸得不均匀一般是什么原因？",
        "fault_ref": "MF-KZE5004 烹饪不均匀时，常见原因是食材放得太多或过于密集，通常应减少份量，必要时中途翻动食材。",
    },
    {
        "model": "MF-KZE5012",
        "short": "5012",
        "capacity": "4.7L",
        "power": "1500W",
        "param_query": "MF-KZE5012 的容量和功率是多少？",
        "param_ref": None,
        "scenario_query": "4.7L 这款更适合两个人还是三个人？",
        "scenario_ref": None,
        "feature_query": "5012 这款不用翻面和普通款差别大吗？",
        "feature_ref": None,
        "usage_query": "MF-KZE5012 第一次使用前需要先清洗哪些部件？",
        "usage_ref": None,
        "clean_query": "5012 清洁时能不能用钢丝球刷烤盘？",
        "clean_ref": "MF-KZE5012 清洁时不建议使用钢丝球、刀片等坚硬物品，以免刮花烤盘和炸桶。",
        "fault_query": "5012 如果烹饪时冒白烟，一般先看什么？",
        "fault_ref": None,
    },
    {
        "model": "MF-KZE5089",
        "short": "5089",
        "capacity": "5L",
        "power": "1550W",
        "param_query": "MF-KZE5089 的容量和功率是多少？",
        "param_ref": None,
        "scenario_query": "5L 带可视窗这款适合几个人用？",
        "scenario_ref": "MF-KZE5089 是 5L 中段容量款，通常更适合两人到三人的日常家用。",
        "feature_query": "5089 的可视窗口平时真的有用吗？",
        "feature_ref": None,
        "usage_query": "MF-KZE5089 首次使用前要不要先清洗炸桶和烤盘？",
        "usage_ref": None,
        "clean_query": "5089 清洁时能不能直接用强力去油剂？",
        "clean_ref": None,
        "fault_query": "5089 如果风扇不转或有异响该怎么办？",
        "fault_ref": None,
    },
    {
        "model": "MF-KZC6054",
        "short": "6054",
        "capacity": "5.5L / 6L",
        "power": "2000W",
        "param_query": "MF-KZC6054 的容量和功率大概是多少？",
        "param_ref": None,
        "scenario_query": "6054 这款适合三四口之家吗？",
        "scenario_ref": None,
        "feature_query": "MF-KZC6054 的双热源和普通款有什么区别？",
        "feature_ref": "MF-KZC6054 的核心差异是上下双热源、免翻面和电子可视，适合更看重加热效率与均匀度的用户。",
        "usage_query": "6054 这种电子控制款怎么选模式？",
        "usage_ref": None,
        "clean_query": "6054 日常清洁时有什么特别要注意的？",
        "clean_ref": None,
        "fault_query": "MF-KZC6054 如果炸锅推不进去一般是什么问题？",
        "fault_ref": "如果 MF-KZC6054 的炸锅推不进去，常见原因是炸锅边缘变形，可以先检查并适当调整边缘结构。",
    },
    {
        "model": "MF-KZC6521",
        "short": "6521",
        "capacity": "高阶大容量段",
        "power": "2000W",
        "param_query": "MF-KZC6521 的额定功率大概是多少？",
        "param_ref": None,
        "scenario_query": "6521 这种高阶款更适合什么样的家庭场景？",
        "scenario_ref": None,
        "feature_query": "MF-KZC6521 的双可视和双热源到底有什么区别？",
        "feature_ref": "MF-KZC6521 主打双可视、上下双热源和电子控制，更适合关注可视化和高阶功能差异的家庭场景。",
        "usage_query": "MF-KZC6521 第一次使用前要做哪些准备？",
        "usage_ref": None,
        "clean_query": "6521 清洁时能不能把烤盘和炸桶一起直接强力刷洗？",
        "clean_ref": None,
        "fault_query": "KZC6521 显示 E1/E2 一般怎么处理？",
        "fault_ref": "MF-KZC6521 显示 E1/E2 时，说明书页面给出的原因是上传感器开路或短路保护，通常应送至指定维修网点处理。",
    },
    {
        "model": "MF-KZE7001",
        "short": "7001",
        "capacity": "7L",
        "power": "1650W",
        "param_query": "7001 的容量和功率大概是多少？",
        "param_ref": None,
        "scenario_query": "7L 这款适合一家几口用？",
        "scenario_ref": None,
        "feature_query": "MF-KZE7001 是双旋钮还是电子面板？",
        "feature_ref": None,
        "usage_query": "7L 这款第一次使用前要做哪些准备？",
        "usage_ref": "第一次使用 MF-KZE7001 前，建议先清洗炸桶等接触食材的部件，再放入食材并通过双旋钮设定参数。",
        "clean_query": "7001 清洁时能不能直接用强力去油清洁剂？",
        "clean_ref": None,
        "fault_query": "7001 风扇不转或者有异响要怎么办？",
        "fault_ref": "MF-KZE7001 风扇不转或者有异响时，常见原因是机体未接通，或电机风扇被异物卡住；应先确认通电状态，必要时联系售后。",
    },
]

for model in models:
    add_entry(
        query=model["param_query"],
        category="商品参数类",
        target_models=[model["model"]],
        difficulty="easy",
        query_scope="single_model",
        model_clarity="explicit",
        usage_tags=["main_candidate"] + (["ragas_candidate"] if model["param_ref"] else []),
        reference=model["param_ref"],
    )
    add_entry(
        query=model["scenario_query"],
        category="适用场景类",
        target_models=[model["model"]],
        difficulty="easy",
        query_scope="single_model",
        model_clarity="weak_feature" if model["capacity"] in {"2.6L", "3L", "4.2L", "4.5L", "4.7L", "5L", "7L"} else "explicit",
        usage_tags=["main_candidate"] + (["ragas_candidate"] if model["scenario_ref"] else []),
        reference=model["scenario_ref"],
    )
    add_entry(
        query=model["feature_query"],
        category="功能差异类",
        target_models=[model["model"]],
        difficulty="medium" if model["model"] in {"MF-KZC6054", "MF-KZC6521"} else "easy",
        query_scope="single_model",
        model_clarity="explicit",
        usage_tags=["main_candidate"] + (["ragas_candidate"] if model["feature_ref"] else []),
        reference=model["feature_ref"],
    )
    add_entry(
        query=model["usage_query"],
        category="使用入门类",
        target_models=[model["model"]],
        difficulty="medium" if model["model"] in {"MF-KZC6054", "MF-KZC6521"} else "easy",
        query_scope="single_model",
        model_clarity="explicit",
        usage_tags=["main_candidate"] + (["ragas_candidate"] if model["usage_ref"] else []),
        reference=model["usage_ref"],
    )
    add_entry(
        query=model["clean_query"],
        category="清洁保养类",
        target_models=[model["model"]],
        difficulty="medium" if model["model"] in {"MF-KZE5089", "MF-KZC6054", "MF-KZC6521", "MF-KZE7001"} else "easy",
        query_scope="single_model",
        model_clarity="explicit" if model["short"] not in {"26E101", "4012", "5012", "5089"} else "weak_feature",
        usage_tags=["main_candidate"] + (["ragas_candidate"] if model["clean_ref"] else []),
        reference=model["clean_ref"],
    )
    add_entry(
        query=model["fault_query"],
        category="常见问题/故障类",
        target_models=[model["model"]],
        difficulty="medium",
        query_scope="single_model",
        model_clarity="explicit",
        usage_tags=["main_candidate"] + (["ragas_candidate"] if model["fault_ref"] else []),
        reference=model["fault_ref"],
    )

compare_entries = [
    ("26E101 和 30E201 都是小容量，主要差别是什么？", ["MF-KZ26E101", "MF-KZ30E201"], "一个人或者宿舍场景里，MF-KZ26E101 更偏 2.6L 极小容量，MF-KZ30E201 则是 3L，空间和份量都会更宽裕一些。"),
    ("一个人用的话，选 2.6L 的 26E101 还是 3L 的 30E201 更合适？", ["MF-KZ26E101", "MF-KZ30E201"], None),
    ("宿舍或小厨房用，4012 和 30E201 哪个更合适？", ["MF-KZE4012", "MF-KZ30E201"], None),
    ("4.2L 的 4012 和 5L 的 5004，家用体验差别大吗？", ["MF-KZE4012", "MF-KZE5004"], None),
    ("5004 和 5012 比，5012 的不用翻面值不值得？", ["MF-KZE5004", "MF-KZE5012"], "如果你更看重不用翻面和顶部双旋钮带来的使用便利，MF-KZE5012 比 MF-KZE5004 有升级价值；如果只要基础 5L 家用功能，5004 会更直接。"),
    ("如果想要 5L 可视窗，是 5089 比 5004 更合适吗？", ["MF-KZE5089", "MF-KZE5004"], "如果你想要 5L 容量并且能看到内部食材状态，MF-KZE5089 通常比 MF-KZE5004 更合适，因为它主打可视窗口和不用翻面。"),
    ("4.5L 可视窗款和 5L 可视窗款该怎么选？", ["MF-KZE459X9BD", "MF-KZE5089"], None),
    ("459X9BD 和 5004 比，最大的区别是不是可视窗？", ["MF-KZE459X9BD", "MF-KZE5004"], "MF-KZE459X9BD 和 MF-KZE5004 的核心差别之一确实是可视窗：459X9BD 便于观察食材状态，而 5004 更偏基础 5L 机械旋钮款。"),
    ("如果我想看得到里面食物状态，459X9BD 和 5004 该选哪个？", ["MF-KZE459X9BD", "MF-KZE5004"], "如果你更看重能直接观察食材状态，通常应优先看 MF-KZE459X9BD，而不是基础旋钮款 MF-KZE5004。"),
    ("5012 和 5089 都主打不用翻面，实际差别大吗？", ["MF-KZE5012", "MF-KZE5089"], None),
    ("7001 和 5089 相比，除了容量还有什么差别？", ["MF-KZE7001", "MF-KZE5089"], "MF-KZE7001 和 MF-KZE5089 除了容量差异外，7001 更偏 7L 大容量双旋钮路线，5089 则是 5L 可视窗中端款，重点在可视窗口和不用翻面。"),
    ("6054 和普通 5L 旋钮款相比，到底值不值得多花钱？", ["MF-KZC6054", "MF-KZE5004"], "如果你更看重上下双热源、免翻面、电子可视和 2000W 大功率，MF-KZC6054 相比普通 5L 旋钮款有明确升级价值；如果只要基础空气炸功能，普通旋钮款通常更够用也更便宜。"),
    ("6054 和 6521 这两款高阶电子款，核心差别是什么？", ["MF-KZC6054", "MF-KZC6521"], "MF-KZC6054 更强调上下双热源、电子可视和 5.5L / 6L 级别容量；MF-KZC6521 则主打双可视、双热源和更高阶的电子控制体验。"),
    ("如果预算够，6521 比 6054 多出来的价值是什么？", ["MF-KZC6521", "MF-KZC6054"], None),
    ("7001 和 6054 一个偏大容量一个偏高阶功能，我该怎么取舍？", ["MF-KZE7001", "MF-KZC6054"], "如果你更看重 7L 大容量和多人份量烹饪，MF-KZE7001 更合适；如果你更看重双热源、电子可视和高阶功能，MF-KZC6054 更值得优先考虑。"),
    ("7001 和 6521 一个偏大容量一个偏高阶功能，我该怎么取舍？", ["MF-KZE7001", "MF-KZC6521"], None),
    ("2.6L 和 4.2L 这两款从日常够用角度怎么选？", ["MF-KZ26E101", "MF-KZE4012"], None),
    ("5089 这种可视窗中端款和 6054 这种双热源款，升级点主要在哪？", ["MF-KZE5089", "MF-KZC6054"], None),
]

for query, targets, ref in compare_entries:
    add_entry(
        query=query,
        category="多型号对比类",
        target_models=targets,
        difficulty="medium",
        query_scope="multi_model",
        model_clarity="explicit",
        usage_tags=["main_candidate"] + (["ragas_candidate"] if ref else []),
        reference=ref,
    )

shared_entries = [
    ("美的空气炸锅支持 7 天无理由退货吗？", "签收后 7 日内、在不影响二次销售的前提下通常可以申请无理由退货；非质量问题退货时，运费一般由买家承担，最终仍需以商品详情页或服务说明为准。"),
    ("你们开发票一般是电子发票吗？", "美的空气炸锅通常默认提供电子发票，具体开具方式仍以商品页服务支持或帮助中心说明为准。"),
    ("这种小家电一般多久发货？偏远地区能送到家吗？", "这类小家电通常在付款后 48 小时内发货；偏远地区或超出派送范围时，可能无法送货上门，需要用户自提。"),
    ("收到外包装破损的机器，签收前应该怎么处理？", "如果外包装明显破损，建议在签收前先拍照留存，并直接拒收或第一时间联系在线客服。"),
    ("空气炸锅坏了之后是 7 天内能退，15 天内能换吗？", None),
    ("如果签收后才发现外观问题，一般该怎么处理？", None),
    ("保修期和保修细则一般去哪里看？", None),
    ("换货的时候能不能直接换成别的型号？", None),
]

for query, ref in shared_entries:
    add_entry(
        query=query,
        category="售后规则类",
        target_models=["shared"],
        difficulty="easy" if ref else "medium",
        query_scope="shared_policy",
        model_clarity="explicit",
        usage_tags=["main_candidate"] + (["ragas_candidate"] if ref else []),
        reference=ref,
    )

explicit_confirmation = [
    ("型号是 MF-KZE7001，这款容量和适用人数大概是多少？", ["MF-KZE7001"]),
    ("型号是 MF-KZC6054，这款是不是电子面板？", ["MF-KZC6054"]),
    ("型号是 MF-KZE5089，这款有没有可视窗？", ["MF-KZE5089"]),
    ("型号是 MF-KZ26E101，这款适合宿舍吗？", ["MF-KZ26E101"]),
    ("型号是 MF-KZE5012，这款是不是主打不用翻面？", ["MF-KZE5012"]),
    ("型号是 MF-KZE4012，这款是不是 4L 左右的小容量？", ["MF-KZE4012"]),
    ("型号是 MF-KZC6521，这款是不是双可视高阶款？", ["MF-KZC6521"]),
    ("型号是 MF-KZE459X9BD，这款能直接看到里面食物状态吗？", ["MF-KZE459X9BD"]),
]

for query, targets in explicit_confirmation:
    add_entry(
        query=query,
        category="型号确认类",
        target_models=targets,
        difficulty="easy",
        query_scope="single_model",
        model_clarity="explicit",
        usage_tags=["model_confirmation_candidate"],
    )

weak_feature_confirmation = [
    ("这款是 2.6L、森墨绿、机械旋钮的小炸锅，应该是哪一个型号？", ["MF-KZ26E101"]),
    ("我买的是 4.2L 白色基础旋钮款，这大概率是哪一款？", ["MF-KZE4012"]),
    ("我这台写的是 4.7L、不用翻面、顶部双旋钮，应该对应哪个型号？", ["MF-KZE5012"]),
    ("5L 带可视窗、双旋控温那款一般是哪一个？", ["MF-KZE5089"]),
    ("我这台是黑色电子款、上下双热源、接近 6L，那是不是 6054？", ["MF-KZC6054"]),
    ("我这台是梨花白、双可视、2000W 的高阶款，对应哪个型号？", ["MF-KZC6521"]),
    ("7L、双旋钮、超大可视窗那台一般是哪款？", ["MF-KZE7001"]),
    ("4.5L 带可视窗但不是 5L 的那款是哪个型号？", ["MF-KZE459X9BD"]),
]

for query, targets in weak_feature_confirmation:
    add_entry(
        query=query,
        category="型号确认类",
        target_models=targets,
        difficulty="medium",
        query_scope="single_model",
        model_clarity="weak_feature",
        usage_tags=["model_confirmation_candidate"],
    )

unconfirmed_confirmation = [
    "我不记得型号了，只记得是 5L 左右的基础旋钮款，能帮我缩小范围吗？",
    "我买的是小容量那台，但忘了是 2.6L 还是 3L，这两种怎么区分？",
    "我只记得是白色方形烤篮的基础款，这种一般是哪几个候选？",
    "我这台能看到里面食物，但我忘了到底是 4.5L、5L 还是 7L，该怎么判断？",
    "我的是电子面板，不是旋钮款，但我不确定是 6054 还是 6521，怎么区分？",
    "我只知道它功率挺大，接近 2000W，这样能判断型号吗？",
    "我手上这台是顶部双旋钮，不知道是 5012 还是 7001，先看什么最有效？",
    "我想按外观判断型号，需要先看哪些信息？",
]

for query in unconfirmed_confirmation:
    add_entry(
        query=query,
        category="型号确认类",
        target_models=["MF-KZ26E101", "MF-KZ30E201", "MF-KZE4012", "MF-KZE459X9BD", "MF-KZE5004", "MF-KZE5012", "MF-KZE5089", "MF-KZC6054", "MF-KZC6521", "MF-KZE7001"],
        difficulty="medium",
        query_scope="single_model",
        model_clarity="unconfirmed",
        usage_tags=["model_confirmation_candidate"],
    )

conflicted_confirmation = [
    ("订单里写的是 MF-KZE5004，但机器上有可视窗，这还是 5004 吗？", ["MF-KZE5004", "MF-KZE459X9BD", "MF-KZE5089", "MF-KZE7001"]),
    ("我以为自己买的是 MF-KZE7001，但机器更像黑色电子面板款，这正常吗？", ["MF-KZE7001", "MF-KZC6054"]),
    ("型号写的是 MF-KZC6521，但我这台却是机械旋钮，这是不是对不上？", ["MF-KZC6521", "MF-KZE5012", "MF-KZE5089", "MF-KZE7001"]),
    ("我一直记成 30E201，但机器看起来更像 2.6L 的森墨绿小款，是不是记错了？", ["MF-KZ30E201", "MF-KZ26E101"]),
    ("订单页像是 KZE5012，但我手上的机器没有顶部双旋钮，这还能对得上吗？", ["MF-KZE5012", "MF-KZE5004", "MF-KZE4012"]),
    ("型号标成 459X9BD，但机器没有可视窗，是不是拿错型号了？", ["MF-KZE459X9BD", "MF-KZE5004", "MF-KZE4012"]),
    ("我记得是 4012，可实际看起来更像 5L 款，这种情况先核对什么？", ["MF-KZE4012", "MF-KZE5004", "MF-KZE5089"]),
    ("6054 这款我以为是 6L 左右电子款，但现在看更像高阶白色双可视，是不是其实是 6521？", ["MF-KZC6054", "MF-KZC6521"]),
]

for query, targets in conflicted_confirmation:
    add_entry(
        query=query,
        category="型号确认类",
        target_models=targets,
        difficulty="medium",
        query_scope="single_model",
        model_clarity="conflicted",
        usage_tags=["model_confirmation_candidate"],
    )

boundary_entries = [
    ("空气炸锅烤红薯一般多少度多少分钟？", ["MF-KZ26E101", "MF-KZ30E201", "MF-KZE4012", "MF-KZE459X9BD", "MF-KZE5004", "MF-KZE5012", "MF-KZE5089", "MF-KZC6054", "MF-KZC6521", "MF-KZE7001"], "使用入门类", "unconfirmed"),
    ("做蛋挞一般要烤多久？", ["MF-KZ26E101", "MF-KZ30E201", "MF-KZE4012", "MF-KZE459X9BD", "MF-KZE5004", "MF-KZE5012", "MF-KZE5089", "MF-KZC6054", "MF-KZC6521", "MF-KZE7001"], "使用入门类", "unconfirmed"),
    ("7001 的滤油盘单买多少钱？", ["MF-KZE7001"], "常见问题/故障类", "explicit"),
    ("你们有没有同系列微波炉适合宿舍？", [], "适用场景类", "unconfirmed"),
    ("这几款里有没有支持联网远程控制的？", ["MF-KZ26E101", "MF-KZ30E201", "MF-KZE4012", "MF-KZE459X9BD", "MF-KZE5004", "MF-KZE5012", "MF-KZE5089", "MF-KZC6054", "MF-KZC6521", "MF-KZE7001"], "功能差异类", "unconfirmed"),
    ("炸鸡翅怎么腌更好吃？", ["MF-KZ26E101", "MF-KZ30E201", "MF-KZE4012", "MF-KZE459X9BD", "MF-KZE5004", "MF-KZE5012", "MF-KZE5089", "MF-KZC6054", "MF-KZC6521", "MF-KZE7001"], "使用入门类", "unconfirmed"),
    ("空气炸锅能不能替代烤箱做蛋糕？", ["MF-KZ26E101", "MF-KZ30E201", "MF-KZE4012", "MF-KZE459X9BD", "MF-KZE5004", "MF-KZE5012", "MF-KZE5089", "MF-KZC6054", "MF-KZC6521", "MF-KZE7001"], "功能差异类", "unconfirmed"),
    ("26E101 和 7001 哪个声音更小？", ["MF-KZ26E101", "MF-KZE7001"], "多型号对比类", "explicit"),
    ("6054 的配件坏了上门维修要多少钱？", ["MF-KZC6054"], "常见问题/故障类", "explicit"),
    ("这些型号哪个二手保值一点？", ["MF-KZ26E101", "MF-KZ30E201", "MF-KZE4012", "MF-KZE459X9BD", "MF-KZE5004", "MF-KZE5012", "MF-KZE5089", "MF-KZC6054", "MF-KZC6521", "MF-KZE7001"], "适用场景类", "unconfirmed"),
]

for query, targets, category, clarity in boundary_entries:
    add_entry(
        query=query,
        category=category,
        target_models=targets,
        difficulty="medium",
        query_scope="cross_boundary",
        model_clarity=clarity,
        usage_tags=["boundary_case"],
    )

main_v3 = [e for e in entries if "main_candidate" in e["usage_tags"] and e["model_clarity"] in {"explicit", "weak_feature"} and e["query_scope"] != "cross_boundary"]
model_confirmation_v2 = [e for e in entries if "model_confirmation_candidate" in e["usage_tags"]]
ragas_v2 = [e for e in entries if "ragas_candidate" in e["usage_tags"] and e["query_scope"] != "cross_boundary"]

if not (110 <= len(entries) <= 140):
    raise ValueError(f"query_bank_v2 count out of range: {len(entries)}")
if not (70 <= len(main_v3) <= 90):
    raise ValueError(f"main_v3 count out of range: {len(main_v3)}")
if not (30 <= len(model_confirmation_v2) <= 40):
    raise ValueError(f"model_confirmation_v2 count out of range: {len(model_confirmation_v2)}")
if not (30 <= len(ragas_v2) <= 36):
    raise ValueError(f"ragas_v2 count out of range: {len(ragas_v2)}")
if len(references) != len(ragas_v2):
    raise ValueError(f"reference count {len(references)} != ragas count {len(ragas_v2)}")

QUERY_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

paths = {
    QUERY_DIR / "air_fryer_midea_query_bank_v2.jsonl": entries,
    QUERY_DIR / "air_fryer_midea_query_set_main_v3.jsonl": main_v3,
    QUERY_DIR / "air_fryer_midea_query_set_model_confirmation_v2.jsonl": model_confirmation_v2,
    QUERY_DIR / "air_fryer_midea_query_set_ragas_v2.jsonl": ragas_v2,
    ANNOTATION_DIR / "main_v3_reference_answers_v1.jsonl": references,
}

for path, data in paths.items():
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(json.dumps({
    "query_bank_v2": len(entries),
    "main_v3": len(main_v3),
    "model_confirmation_v2": len(model_confirmation_v2),
    "ragas_v2": len(ragas_v2),
    "references": len(references),
}, ensure_ascii=False))

