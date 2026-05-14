"""临时辅助:把 per_sample.jsonl 中某个 ablation 的检索结果与 gold 并排打印。"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/rq1_real/per_sample.jsonl"
target = sys.argv[2] if len(sys.argv) > 2 else "R5_full"
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
for r in rows:
    if r.get("ablation") != target:
        continue
    qid = r["query_id"]
    gold = r["gold_ids"]
    top5 = r["retrieved_ids"][:5]
    print(f"{qid}: gold={gold}, top5={top5}")
