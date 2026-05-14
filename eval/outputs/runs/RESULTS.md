# 首轮评估结果(smoke + 初步真实数据)

> ⚠️ 这是工程链路验证 + 真实数据的首轮快照,**不是论文最终结果**。
> 数据集是 10 条 QA(其中 1 条 OOS),gold_chunk_ids 是用宽语义 LIKE/ILIKE 自动打标的,
> 论文使用前必须人工 review 收窄,并扩到 200+ 题。
>
> 模型与时间见各 `manifest.json`;benchmark SHA256 已写入 manifest 用于复现。

---

## RQ1 — 检索 ablation(top_k=10,9 条有效样本)

`runs/rq1_v3/aggregate.csv`

| ablation | n | Recall@1 | Recall@5 | Recall@10 | Hit@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|
| R1_vector_only | 9 | 0.003 | 0.003 | 0.003 | 0.111 | 0.111 | 0.025 |
| R2_vector_overfetch | 9 | 0.003 | 0.003 | 0.003 | 0.111 | 0.111 | 0.025 |
| R3_plus_keyword | 9 | 0.014 | 0.014 | 0.014 | 0.111 | 0.111 | 0.028 |
| R4_plus_rrf | 9 | **0.017** | **0.017** | **0.017** | **0.222** | **0.222** | **0.053** |
| R5_full | 9 | 0.017 | 0.017 | 0.017 | 0.222 | 0.222 | 0.053 |

**观察:**
- **Hit@5 / MRR**:R4/R5 较 R1/R2 翻倍(0.111 → 0.222)
- **nDCG@10**:R4/R5 ≈ R1/R2 的 2 倍
- 当前 gold 集对 Q006/Q007/Q009 被关键词扩到 30+ 个,Recall 上限被压低 → 论文需人工筛 gold
- Reranker(R5 vs R4)在本数据集上未见额外收益,可能 query 不复杂 / 候选集已足够好

---

## RQ2 — 端到端答案质量(RAGAS,judge=DeepSeek)

`runs/rq2_v2/aggregate.csv`

| ablation | n | Faithfulness | Answer Relevance | Context Relevance |
|---|---|---|---|---|
| A0 plain LLM | 10 | 0.100 | 0.990 | 0.000 |
| A1 RAG only | 10 | **0.365** | 0.690 | 0.020 |
| A2 EduAgent full | 10 | **0.401** | 0.550 | 0.020 |

**核心发现:**
- **Faithfulness 0.10 → 0.36 → 0.40**:RAG 带来的忠实度提升非常清晰,**这是 RAG 对教学场景价值的直接证据**
- **Answer Relevance 0.99 → 0.69 → 0.55** 反向下降:judge 认为 RAG 答案"没那么直接"(更克制 / 带引用 / 偏长),这是个值得讨论的 trade-off
- **Context Relevance 仅 0.02**:与 RQ1 低绝对值一致,说明 retriever 在该语料上还有相当大的优化空间(向量层只命中 11% Hit@5)
- A3_baseline_llm 因未配 `BASELINE_*` env 跳过,不影响主线

---

## RQ4 — 流式接口性能(SSE TTFT / E2E)

`runs/rq4_v1/aggregate.csv` (concurrencies=1,3,5,每条 query 重复 2 次)

| 并发 | 总/成功/失败 | wall(s) | QPS | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | E2E p50 (ms) | E2E p95 (ms) | E2E p99 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 20/20/0 | 92.2 | 0.22 | **594.8** | 1074.0 | 1602.3 | 4289.1 | 6759.6 | 7636.1 |
| 3 | 20/12/8 | 28.2 | 0.43 | 1056.6 | 2415.4 | 2415.4 | 5434.6 | 6454.0 | 6454.0 |
| 5 | 20/12/8 | 21.0 | 0.57 | 1399.1 | 2347.0 | 2347.0 | 6222.0 | 9524.6 | 9524.6 |

**观察:**
- **并发=1**:全部 20 个请求成功,TTFT p50 < 600ms,p99 < 1.7s,**满足课堂实时交互**
- **并发=3/5**:成功率降到 60%,失败来自 ModelScope 端点对突发并发的限流,**不是后端编排瓶颈**
- QPS 上升但收益递减(0.22 → 0.43 → 0.57),说明在外部 LLM 后端的实际约束下,QPS 上限主要由上游决定
- 论文里可单写一组 "private LLM endpoint vs free-tier endpoint" 对比来分离这部分

---

## 给论文的下一步

1. **扩 QA 基准到 200+**,按 type/difficulty 分层
2. **人工 review gold_chunk_ids**(当前自动打标只是 ceiling 估算)
3. **跑 ≥ 3 次重复求均值 ± 标准差**(RQ4 已支持 `--repeats-per-query`,RQ1/RQ2 需要外层包装)
4. **加 A3 基线**:`BASELINE_API_KEY=`、`BASELINE_BASE_URL=`、`BASELINE_MODEL_NAME=Qwen/Qwen3-8B`(同家弱模型),回答 RQ2 的"是 RAG 重要还是模型规模重要"
5. **私有 LLM 端点**(vLLM 自建)对 RQ4 重跑一次,把"系统瓶颈"与"上游瓶颈"切干净
6. **统计检验**:对 A0/A1/A2 的 Faithfulness 做 Wilcoxon signed-rank,看 p < 0.05 是否成立(N=10 偏小,需扩到 ≥ 30)

---

## 对应 manifest

- RQ1: `runs/rq1_v3/manifest.json` — benchmark SHA `0aac9c…`(以实际为准),timestamp_utc 见文件
- RQ2: `runs/rq2_v2/manifest.json` — judge=`deepseek-v4-pro` @ DeepSeek
- RQ4: `runs/rq4_v1/manifest.json` — base_url=http://localhost:8000
