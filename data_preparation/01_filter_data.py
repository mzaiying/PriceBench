"""
新 Bench 数据筛选脚本
从 cases.json (14,951条) 筛选双价格样本，输出到 data/bench_raw.json
"""
import json
import statistics
from collections import Counter
from pathlib import Path

# 路径
ROOT = Path(__file__).parent.parent
CASES_FILE = ROOT / "data" / "cases.json"
OUTPUT_FILE = Path(__file__).parent / "data" / "bench_raw.json"

# ========== Step 1: 加载原始数据 ==========
with open(CASES_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
print(f"[1] 加载原始数据: {len(raw_data)} 条")

# ========== Step 2: 过滤无效条目 ==========
SKIP_NAMES = {
    '序号', '合计', '小计', '总计', '总价', '费用合计', '项目总投资', '投资总计',
    '费用类别', '项目名称', '建设内容', '一级', '二级', '三级', '编号', '备注',
    '预备费', '基本预备费', '涨价预备费', '建设期利息', '铺底流动资金',
    '不可预见费', '项目管理费', '招标代理费', '工程监理费', '测试验收费',
    '第三方检测费', '培训费', '差旅费'
}

valid = []
for d in raw_data:
    name = (d.get("item_name") or "").strip()
    if not name or len(name) < 2:
        continue
    if name.replace(".", "").replace(" ", "").replace("、", "").isdigit():
        continue
    if name in SKIP_NAMES:
        continue
    valid.append(d)
print(f"[2] 过滤无效条目: {len(valid)} 条有效 (去掉 {len(raw_data) - len(valid)} 条)")

# ========== Step 3: 筛选双价格样本 ==========
bench_samples = []
sample_id = 0

for d in valid:
    # 获取原始价格和调整后价格
    orig_total = d.get("total_original")
    redu_total = d.get("total_reduced")
    orig_price = d.get("price_original")
    redu_price = d.get("price_reduced")

    # 优先使用 total，其次使用 unit price
    orig = orig_total if orig_total and float(orig_total) > 0 else orig_price
    redu = redu_total if redu_total and float(redu_total) > 0 else redu_price

    if not orig or not redu:
        continue
    orig = float(orig)
    redu = float(redu)
    if orig <= 0 or redu <= 0:
        continue

    # 计算核减率
    adjustment_rate = round((orig - redu) / orig * 100, 2)

    # 判断方向
    if abs(adjustment_rate) < 0.01:
        direction = "unchanged"
    elif adjustment_rate > 0:
        direction = "reduced"
    else:
        direction = "increased"

    sample_id += 1
    sample = {
        "sample_id": f"S{sample_id:04d}",
        "item_name": (d.get("item_name") or "").strip(),
        "spec_original": (d.get("spec_original") or "").strip(),
        "spec_reduced": (d.get("spec_reduced") or "").strip(),
        "unit": (d.get("unit") or "").strip(),
        "qty_original": d.get("qty_original"),
        "qty_reduced": d.get("qty_reduced"),
        "unit_price_original": d.get("price_original"),
        "unit_price_reduced": d.get("price_reduced"),
        "total_original": orig,
        "total_reduced": redu,
        "adjustment_rate": adjustment_rate,
        "direction": direction,
        "change_type": d.get("change_type", ""),
        "category": d.get("category", ""),
        "sheet_name": d.get("sheet_name", ""),
        "project_name": d.get("project_name", ""),
        "remark": (d.get("remark") or "").strip(),
    }
    bench_samples.append(sample)

print(f"[3] 双价格样本: {len(bench_samples)} 条")

# ========== Step 4: 数据质量检查 ==========
print(f"\n{'='*60}")
print(f"数据质量检查")
print(f"{'='*60}")

# 4.1 方向分布
dirs = Counter(s["direction"] for s in bench_samples)
print(f"\n方向分布:")
for d, c in dirs.most_common():
    print(f"  {d}: {c} ({c/len(bench_samples)*100:.1f}%)")

# 4.2 费用类别分布
cats = Counter(s["category"] for s in bench_samples)
print(f"\n费用类别分布 ({len(cats)} 类):")
for c, n in cats.most_common():
    print(f"  {c}: {n} ({n/len(bench_samples)*100:.1f}%)")

# 4.3 项目分布
projs = Counter(s["project_name"][:35] for s in bench_samples)
print(f"\n项目分布 ({len(projs)} 个项目):")
for p, n in projs.most_common():
    print(f"  {p}: {n}")

# 4.4 核减率统计
rates = [s["adjustment_rate"] for s in bench_samples]
print(f"\n核减率统计:")
print(f"  Mean: {statistics.mean(rates):.2f}%")
print(f"  Median: {statistics.median(rates):.2f}%")
print(f"  Stdev: {statistics.stdev(rates):.2f}%")
print(f"  Min: {min(rates):.2f}%, Max: {max(rates):.2f}%")

# 4.5 异常值检查
extreme = [s for s in bench_samples if abs(s["adjustment_rate"]) > 200]
print(f"\n极端核减率 (|rate| > 200%): {len(extreme)} 条")
for s in extreme[:5]:
    print(f"  {s['sample_id']}: {s['item_name'][:20]} rate={s['adjustment_rate']:.1f}% "
          f"orig={s['total_original']:.1f} redu={s['total_reduced']:.1f}")

# 4.6 字段完整性
has_spec = sum(1 for s in bench_samples if s["spec_original"] or s["spec_reduced"])
has_qty = sum(1 for s in bench_samples if s["qty_original"] is not None)
has_unit_price = sum(1 for s in bench_samples if s["unit_price_original"] is not None)
has_remark = sum(1 for s in bench_samples if s["remark"])
print(f"\n字段完整性:")
print(f"  有规格描述: {has_spec}/{len(bench_samples)} ({has_spec/len(bench_samples)*100:.1f}%)")
print(f"  有数量: {has_qty}/{len(bench_samples)} ({has_qty/len(bench_samples)*100:.1f}%)")
print(f"  有单价: {has_unit_price}/{len(bench_samples)} ({has_unit_price/len(bench_samples)*100:.1f}%)")
print(f"  有专家备注: {has_remark}/{len(bench_samples)} ({has_remark/len(bench_samples)*100:.1f}%)")

# 4.7 去掉极端异常值后的清洁数据
clean = [s for s in bench_samples if abs(s["adjustment_rate"]) <= 200]
print(f"\n去掉极端异常值后: {len(clean)} 条 (去掉 {len(bench_samples) - len(clean)} 条)")

# ========== Step 5: 输出 ==========
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(clean, f, ensure_ascii=False, indent=2)
print(f"\n[5] 已输出到: {OUTPUT_FILE}")
print(f"    最终样本数: {len(clean)} 条")

# 展示前 3 条样本
print(f"\n{'='*60}")
print(f"样本示例（前 3 条）")
print(f"{'='*60}")
for s in clean[:3]:
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print("---")
