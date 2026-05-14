import json, sys, collections
path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
c = collections.Counter(r["ablation"] for r in rows)
for k, v in c.items():
    print(f"{k}: {v}")
