# EduAgent 微调指南

本文档描述如何利用系统收集的用户反馈数据，对基座大模型进行微调，以提升教育场景下的回答质量。

---

## 一、反馈数据导出

系统通过 `feedbacks` 表收集用户对每次对话的评分反馈。可以通过以下 SQL 查询导出训练数据：

### 导出所有反馈数据

```sql
SELECT
    f.id,
    f.prompt,
    f.response,
    f.rating,
    f.comment,
    f.created_at
FROM feedbacks f
ORDER BY f.created_at;
```

### 导出正面反馈（用于 SFT）

```sql
SELECT prompt, response
FROM feedbacks
WHERE rating = 1
ORDER BY created_at;
```

### 导出配对数据（用于 DPO）

```sql
SELECT
    pos.prompt AS prompt,
    pos.response AS chosen,
    neg.response AS rejected
FROM feedbacks pos
JOIN feedbacks neg ON pos.prompt = neg.prompt
WHERE pos.rating = 1 AND neg.rating = -1;
```

---

## 二、SFT 数据格式（Alpaca 格式）

将正面反馈数据转换为 Alpaca 格式的 JSON 文件：

```json
[
  {
    "instruction": "请解释牛顿第二定律的含义及其在日常生活中的应用。",
    "input": "",
    "output": "牛顿第二定律（F=ma）表明...[1]\n\n**参考来源：**\n[1] 物理教材.md > 第三章 > 3.2 牛顿第二定律"
  },
  {
    "instruction": "什么是光合作用？请用简单的语言解释。",
    "input": "",
    "output": "光合作用是植物利用阳光将二氧化碳和水转化为葡萄糖和氧气的过程...[1][2]"
  }
]
```

**字段说明：**
- `instruction`：用户的提问（对应 `feedbacks.prompt`）
- `input`：额外输入信息（通常为空字符串）
- `output`：AI 的优质回答（对应 `feedbacks.response`，仅取 `rating=1` 的数据）

**数据导出脚本示例：**

```python
import json
import psycopg2

conn = psycopg2.connect("postgresql://postgres:postgres@localhost:5432/eduagent")
cur = conn.cursor()
cur.execute("SELECT prompt, response FROM feedbacks WHERE rating = 1")

data = []
for prompt, response in cur.fetchall():
    data.append({
        "instruction": prompt,
        "input": "",
        "output": response,
    })

with open("sft_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"导出 {len(data)} 条 SFT 数据")
```

---

## 三、DPO 数据格式

DPO（Direct Preference Optimization）需要同一 prompt 下的正负样本对：

```json
[
  {
    "prompt": "请解释牛顿第二定律。",
    "chosen": "牛顿第二定律（F=ma）是经典力学的核心定律之一...",
    "rejected": "牛顿第二定律就是 F=ma。"
  }
]
```

**字段说明：**
- `prompt`：用户的提问
- `chosen`：用户评为好的回答（`rating=1`）
- `rejected`：用户评为差的回答（`rating=-1`）

---

## 四、LLaMA-Factory 安装与配置

### 1. 环境准备

```bash
# 克隆 LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 创建虚拟环境
conda create -n llama_factory python=3.11
conda activate llama_factory

# 安装依赖
pip install -e ".[torch,metrics]"
```

### 2. 配置数据集

将 SFT 数据文件放到 `data/` 目录下，并在 `data/dataset_info.json` 中注册：

```json
{
  "eduagent_sft": {
    "file_name": "sft_data.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "eduagent_dpo": {
    "file_name": "dpo_data.json",
    "formatting": "alpaca",
    "ranking": true,
    "columns": {
      "prompt": "prompt",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
}
```

---

## 五、基于 Qwen2.5-7B 的 LoRA 微调

### 1. SFT 微调命令

```bash
llamafactory-cli train \
  --stage sft \
  --do_train true \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset eduagent_sft \
  --template qwen \
  --finetuning_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj \
  --output_dir saves/qwen2.5-7b-eduagent-sft \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_train_epochs 3 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --logging_steps 10 \
  --save_steps 100 \
  --bf16 true \
  --max_length 2048 \
  --report_to tensorboard
```

### 2. DPO 微调命令

```bash
llamafactory-cli train \
  --stage dpo \
  --do_train true \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_name_or_path saves/qwen2.5-7b-eduagent-sft \
  --dataset eduagent_dpo \
  --template qwen \
  --finetuning_type lora \
  --lora_rank 16 \
  --lora_alpha 32 \
  --output_dir saves/qwen2.5-7b-eduagent-dpo \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 1 \
  --lr_scheduler_type cosine \
  --bf16 true \
  --max_length 2048 \
  --dpo_beta 0.1 \
  --report_to tensorboard
```

### 3. 合并 LoRA 权重（可选）

```bash
llamafactory-cli export \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --adapter_name_or_path saves/qwen2.5-7b-eduagent-dpo \
  --template qwen \
  --finetuning_type lora \
  --export_dir models/qwen2.5-7b-eduagent-merged \
  --export_size 2 \
  --export_legacy_format false
```

---

## 六、微调后模型部署

### 方式一：使用 vLLM 部署

```bash
pip install vllm

# 使用合并后的模型
python -m vllm.entrypoints.openai.api_server \
  --model models/qwen2.5-7b-eduagent-merged \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 4096 \
  --tensor-parallel-size 1

# 或使用 LoRA 适配器（无需合并）
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --enable-lora \
  --lora-modules eduagent=saves/qwen2.5-7b-eduagent-dpo \
  --host 0.0.0.0 \
  --port 8001
```

### 方式二：使用 Ollama 部署

```bash
# 1. 创建 Modelfile
cat > Modelfile << 'EOF'
FROM models/qwen2.5-7b-eduagent-merged
TEMPLATE """{{ .System }}
{{ .Prompt }}"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# 2. 创建 Ollama 模型
ollama create eduagent -f Modelfile

# 3. 运行模型
ollama serve  # 默认端口 11434
```

### 接入 EduAgent 系统

微调模型部署后，只需修改 `.env` 文件中的 LLM 配置即可无缝接入：

```env
# vLLM 部署
LLM_BASE_URL=http://localhost:8001/v1
LLM_MODEL_NAME=models/qwen2.5-7b-eduagent-merged
LLM_API_KEY=not-needed

# 或 Ollama 部署
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL_NAME=eduagent
LLM_API_KEY=not-needed
```

重启 backend 服务后即可生效，无需修改任何代码。

---

## 七、注意事项

1. **数据质量**：微调效果强依赖数据质量，建议积累至少 500+ 条高质量反馈后再开始微调
2. **评估指标**：微调前后可使用 BLEU、ROUGE 等指标对比效果
3. **显存要求**：Qwen2.5-7B 的 LoRA 微调大约需要 16GB+ 显存（单卡 A100/4090）
4. **迭代优化**：建议定期导出新的反馈数据，持续迭代微调
