# EduAgent 评估工程(RQ1 + RQ2 + RQ4)

面向论文发表的可复现实验框架。覆盖三大研究问题:

| RQ | 主题 | 指标 | Runner |
|---|---|---|---|
| **RQ1** | 混合检索 ablation | Recall@k / MRR / nDCG@10 | `runners/run_retrieval.py` |
| **RQ2** | 端到端答案质量 | Faithfulness / Answer Relevance / Context Relevance (RAGAS) | `runners/run_answer.py` |
| **RQ4** | 流式接口性能 | TTFT / E2E p50/p95/p99 / QPS | `runners/run_perf.py` |

---

## 目录结构

```
eval/
├── README.md
├── benchmarks/
│   ├── qa_v1.jsonl              # 10 条示例 QA(请按 schema 扩到 200+)
│   ├── schema.py                # QAItem 定义 + load/dump JSONL
│   └── builder/
│       └── auto_extract.py      # 从教材 .md 半自动出题
├── configs/
│   └── ablations.py             # RQ1_ABLATIONS / RQ2_ABLATIONS 清单
├── metrics/
│   ├── retrieval.py             # Recall@k / MRR / nDCG / Hit@k
│   ├── ragas_eval.py            # LLM-as-judge 三件套
│   ├── perf_stats.py            # 百分位 / QPS
│   └── report.py                # CSV / Markdown / LaTeX 表格输出
├── runners/
│   ├── run_retrieval.py         # RQ1
│   ├── run_answer.py            # RQ2
│   └── run_perf.py              # RQ4
└── outputs/
    └── runs/{timestamp}/        # 每次运行的原始日志 / per-sample / aggregate
```

---

## 一、准备:扩展基准数据集

`benchmarks/qa_v1.jsonl` 内 10 条样例的 `gold_chunk_ids` 是空的;你需要:

1. 通过 `POST /api/knowledge/upload` 把教材导入,查 PG 拿到 `knowledge_chunks.id`
2. 对每条 query 标注其 gold chunk(可工具辅助:用 `retrieve_detailed` 跑一次,人工挑出真正相关的)
3. 用 schema.py 校验:
   ```bash
   python -c "from eval.benchmarks.schema import load_jsonl; print(len(load_jsonl('eval/benchmarks/qa_v1.jsonl')))"
   ```

或半自动构建(强烈建议手工 review):

```bash
# 在 backend 容器或装好后端依赖的 venv 内
python -m eval.benchmarks.builder.auto_extract /path/to/教材.md \
  --output eval/benchmarks/candidates.jsonl --max-sections 10
```

候选输出 `candidates.jsonl` 是 LLM 出的草稿,需要人工:
- 检查 query 合理性、改写不自然的措辞
- 填 `gold_chunk_ids`(从数据库查)
- 标注 `difficulty` / `needs_tool`

**论文复现性要求:数据集发布时附 SHA256,本仓库的 runner 已自动记录到 manifest。**

---

## 二、RQ1:检索 ablation

5 个配置(`configs/ablations.py::RQ1_ABLATIONS`):

| 编号 | 名称 | Over-fetch | Keyword | RRF | Reranker | 取哪一阶段 top_k |
|---|---|---|---|---|---|---|
| R1 | vector_only | 1x | ✗ | ✗ | ✗ | vector |
| R2 | vector_overfetch | 4x | ✗ | ✗ | ✗ | vector |
| R3 | plus_keyword | 4x | ✓ | ✗ | ✗ | keyword |
| R4 | plus_rrf | 4x | ✓ | ✓ | ✗ | fused |
| R5 | full(默认) | 4x | ✓ | ✓ | ✓ | final |

执行:

```bash
docker compose exec backend python -m eval.runners.run_retrieval \
  --benchmark eval/benchmarks/qa_v1.jsonl \
  --output-dir eval/outputs/runs/rq1_$(date +%Y%m%d_%H%M%S) \
  --top-k 10
```

输出:
- `per_sample.jsonl` — 每条样本 × 每个 ablation 的 retrieved_ids / gold_ids
- `aggregate.csv` / `aggregate.md` / `aggregate.tex` — 5 行汇总
- `manifest.json` — 数据集 hash / 时间 / 参数

---

## 三、RQ2:端到端答案质量

4 个 ablation(`configs/ablations.py::RQ2_ABLATIONS`):

| 编号 | 名称 | RAG | 工具 | LLM |
|---|---|---|---|---|
| A0 | plain_llm | ✗ | ✗ | LLM_* |
| A1 | rag_only | ✓ | ✗ | LLM_* |
| A2 | eduagent_full | ✓ | ✓ | LLM_* |
| A3 | baseline_llm | ✓ | ✓ | BASELINE_*(需配 env) |

> 注:当前 A2/A3 的"工具"字段更多是占位 — 短答题路径并不会触发工具调用。论文中可侧重 A0 vs A1 vs A3 来证明 RAG 与模型选择的影响,或扩展 query 设计以触发工具。

环境变量:

```bash
# 被测主模型(已有,通常无需改)
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL_NAME=...

# Judge:务必比被测更强,避免自评偏置
JUDGE_API_KEY=sk-...
JUDGE_BASE_URL=https://api.openai.com/v1
JUDGE_MODEL_NAME=gpt-4o

# 可选:基线对比
BASELINE_API_KEY=...
BASELINE_BASE_URL=...
BASELINE_MODEL_NAME=Qwen/Qwen3-8B
```

执行:

```bash
docker compose exec backend python -m eval.runners.run_answer \
  --benchmark eval/benchmarks/qa_v1.jsonl \
  --output-dir eval/outputs/runs/rq2_$(date +%Y%m%d_%H%M%S) \
  --top-k 5
```

输出包含 RAGAS 三件套均值 + 每条样本明细。

---

## 四、RQ4:性能

测 SSE 流式接口的真实 TTFT / E2E 延迟与吞吐。**默认关闭 mermaid / wechat / 外部工具**,只跑核心 RAG + LLM 流。

执行(从宿主机或任意有网络的位置打到 backend):

```bash
python -m eval.runners.run_perf \
  --base-url http://localhost:8000 \
  --benchmark eval/benchmarks/qa_v1.jsonl \
  --concurrencies 1,5,10,20 \
  --repeats-per-query 3 \
  --output-dir eval/outputs/runs/rq4_$(date +%Y%m%d_%H%M%S)
```

输出:
- `per_request.jsonl` — 每个请求的 timestamp 链
- `aggregate.csv` — 各并发档 × {TTFT/E2E p50/p95/p99, QPS}
- `aggregate.md` / `.tex`

> 论文里可单独写一组"冷启动 vs 热启动":两次跑同一并发档,首次为冷,第二次为热,比 mean 差。

---

## 五、把结果搬进论文

每个 runner 都同时产出:
- `aggregate.csv` — 直接 import 到 pandas / Excel
- `aggregate.md`  — 贴回项目 README / 实验记录
- `aggregate.tex` — `\include` 到 LaTeX 主文档

manifest.json 内容(数据集 hash / 模型版本 / commit SHA)直接抄进 paper 的 reproducibility 段落。

---

## 六、复现性硬要求

| 项 | 已实现 |
|---|---|
| temperature=0 评估 | ✅(generator / judge 都已固定) |
| 数据集 hash 写入 manifest | ✅ |
| 模型 / endpoint 写入 manifest | ✅ |
| 时间戳(UTC) | ✅ |
| 多次重复取均值 | ⚠️ RQ4 已支持 `--repeats-per-query`;RQ1/RQ2 需外层脚本跑 N 次再求均 |
| commit SHA 自动记录 | ⏳ 待加(暂在 manifest 里手工补) |

---

## 七、依赖

`run_retrieval.py` / `run_answer.py` 需要在装好后端依赖的环境内跑(import `app.*`),最简单是在 `eduagent-backend` 容器内:

```bash
docker compose exec backend bash -c "cd /app && python -m eval.runners.run_retrieval ..."
```

`run_perf.py` 只依赖 `httpx`,宿主机直接跑即可。

---

## 八、首轮已跑数据

`outputs/runs/RESULTS.md` 汇总了 RQ1 + RQ2 + RQ4 三组的 smoke-grade 真实结果(N=10,judge=DeepSeek):

- **RQ1**:R4/R5 完整混合检索 Hit@5 是 R1 向量-only 的 **2×**
- **RQ2**:**Faithfulness 0.10 → 0.36 → 0.40**(plain → RAG → EduAgent),RAG 的教学忠实度收益清晰
- **RQ4**:并发=1 时 TTFT p99 < 1.7s,并发≥3 后失败率来自上游 LLM 限流(非系统瓶颈)

每个 run 都附 manifest.json(SHA / 时间 / 模型),论文可复现。

---

## 九、扩展点

- 增加 ablation:编辑 `configs/ablations.py`,加新条目即可,不用改 runner
- 替换 judge 实现:`metrics/ragas_eval.py` 三个函数都是独立的,改 prompt 或换模型互不影响
- 加新指标:在 `metrics/` 新建模块,runner 把新字段塞进 summary_rows 自动出现在 CSV/Markdown
- 工具调用评估(RQ3):待补 `runners/run_tools.py`,可参考 run_answer 的结构
