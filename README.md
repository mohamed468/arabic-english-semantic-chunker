<div align="center">
  
  <img src="https://img.shields.io/badge/Language-Python_3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Model-Qwen_2.5_1.5B-FF6B35?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/Tech-LoRA_Fine--Tuning-8A2BE2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Serving-vLLM_0.7.2-00C7B7?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Load_Testing-Locust-5B8C5A?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-Apache_2.0-green?style=for-the-badge"/>
  
  # Arabic-English Semantic Chunker (LLM-Based)
  
  **A Multilingual Semantic Text Segmentation System Powered by Large Language Models**
  
  *Fine-tuned Qwen 2.5 1.5B model using LoRA for Semantic Chunking of Arabic, English, and Code-Switched texts—featuring an end-to-end pipeline from synthetic data generation to production-grade vLLM serving.*
</div>

---

## Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Hardware Requirements](#-hardware-requirements)
- [Installation](#-installation)
- [Usage & API Examples](#-usage--api-examples)
- [Performance & Load Testing](#-performance--load-testing)
- [Engineering Insights & Production Analysis](#-engineering-insights--production-analysis)
  
---

## Project Overview
The **Arabic Semantic Chunker** represents a complete LLMOps lifecycle. It solves a critical issue in modern Retrieval-Augmented Generation (RAG) systems: traditional text splitters (chunking by character count or punctuation) destroy semantic context, especially in complex languages like Arabic. 

This project leverages a fine-tuned LLM capable of understanding semantic boundaries, ensuring that documents are split into logically coherent, self-contained chunks without losing contextual integrity.

---

## Key Features
| Feature | Description |
|---|---|
| ** Multilingual Support** | Processes Modern Standard Arabic, English, and technical code-switched texts seamlessly. |
| ** Structured Outputs** | Guarantees syntactically valid JSON outputs strictly adhering to a Pydantic schema. |
| ** High-Throughput Serving** | Deployed via vLLM with dynamic LoRA adapter loading for production readiness. |
| ** Quality Assurance** | Automated filtering of anomalies (e.g., hallucinatory characters) post-training. |
| ** Load Tested** | Stress-tested using Locust, simulating high-concurrency environments. |
| ** Resource Efficient** | Trains and serves a 1.5B parameter model effectively on a single 15GB VRAM GPU (T4). |

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                    End-to-End LLMOps Pipeline                   │
└─────────────────────────────────────────────────────────────────┘
  Phase 1                 Phase 2                  Phase 3
┌──────────────┐        ┌──────────────┐        ┌──────────────────┐
│ Synthetic    │        │ LoRA         │        │ vLLM Production  │
│ Data Gen     │ ──────►│ Fine-Tuning  │ ──────►│ Serving          │
│              │        │              │        │                  │
│ • Faker      │        │ • Unsloth    │        │ • REST API       │
│ • Gemini     │        │ • Qwen2.5    │        │ • Dynamic LoRA   │
│ • 2700+ rows │        │ • 1000 steps │        │ • Port 8000      │
└──────────────┘        └──────────────┘        └────────┬─────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────┐
                                              │ Stress Testing   │
                                              │ (Locust)         │
                                              │                  │
                                              │ • 5-20 Users     │
                                              │ • 60 Seconds     │
                                              │ • Tokens/sec     │
                                              └──────────────────┘
```

### Internal Processing Flow

```text
Raw Text (Arabic/English/Mixed)
        │
        ▼
┌───────────────────┐
│  Chat Template    │  ← System Message + Strict Schema constraints
│  (Qwen Format)    │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Qwen 2.5 1.5B    │
│  + LoRA Adapter   │  ← Mo-Abdelfattah/arabic-semantic-chunker-qwen1.5b
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  JSON Output      │  ← Post-processing & validation via json_repair
│  (Pydantic)       │
└────────┬──────────┘
         │
         ▼
{
  "original_text_length": 342,
  "semantic_chunks": [
    "First independent semantic chunk...",
    "Second semantic chunk..."
  ]
}
```

---

## Repository Structure

```text
arabic-semantic-chunker/
│
├── notebooks/
│   ├── 01_data_generation.ipynb      # Synthetic data generation via Gemini & Faker
│   ├── 02_fine_tuning.ipynb          # SFT pipeline using Unsloth & LoRA
│   ├── 03_evaluation.ipynb           # Model evaluation and output sanitization
│   └── 04_deployment_load_test.ipynb # vLLM serving and Locust load testing
│
├── scripts/
│   ├── data/
│   │   ├── generate_synthetic.py     
│   │   └── validate_dataset.py       
│   ├── serving/
│   │   ├── start_server.sh           # vLLM initialization script
│   │   └── locustfile.py             # Locust stress testing scenarios
│   └── evaluation/
│       └── run_eval.py               
│
├── configs/
│   ├── model_config.yaml             
│   └── serving_config.yaml           
│
├── requirements.txt               
└── README.md
```

---

## Hardware Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **GPU VRAM** | 12 GB | 15 GB (e.g., T4, L4, A10g) |
| **RAM** | 12 GB | 16 GB+ |
| **Python** | 3.10 | 3.12 |
| **CUDA** | 12.1 | 12.4+ |
| **OS** | Linux | Ubuntu 22.04 |

---

## Installation

**Step 1: Clone the repository**
```bash
git clone https://github.com/Mo-Abdelfattah/arabic-semantic-chunker.git
cd arabic-semantic-chunker
```

**Step 2: Install Unsloth**

Unsloth must be installed first and separately. Refer to the [official Unsloth repository](https://github.com/unslothai/unsloth) for the version matching your CUDA setup. For Google Colab:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**Step 3: Install remaining dependencies**
```bash
pip install -r requirements.txt
```

> **Note:** `numpy<2` is pinned in `requirements.txt` to ensure compatibility with vLLM and CUDA 12. Installing Unsloth first avoids import-order conflicts with `torch` and `transformers`.

**Step 4: Spin up the vLLM Server**
```bash
BASE_MODEL="unsloth/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH="Mo-Abdelfattah/arabic-semantic-chunker-qwen1.5b"

nohup vllm serve "$BASE_MODEL" \
  --dtype=half \
  --gpu-memory-utilization 0.8 \
  --max-model-len 2048 \
  --max-lora-rank 64 \
  --enable-lora \
  --lora-modules arabic-chunker="$ADAPTER_PATH" \
  > nohup.out 2>&1 &
```

---

### Step 4: Spin up the vLLM Server
    BASE_MODEL="unsloth/Qwen2.5-1.5B-Instruct"
    ADAPTER_PATH="Mo-Abdelfattah/arabic-semantic-chunker-qwen1.5b"

    nohup vllm serve "$BASE_MODEL" \
      --dtype=half \
      --gpu-memory-utilization 0.8 \
      --max-model-len 2048 \
      --max-lora-rank 64 \
      --enable-lora \
      --lora-modules arabic-chunker="$ADAPTER_PATH" \
      > nohup.out 2>&1 &

---

## Usage & API Examples

### Python REST API Call

    import requests
    from transformers import AutoTokenizer
    import json

    tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct")

    input_text = "أسلوب إديسون في العمل كان يعتمد على Trial and Error. كان يمتلك فريقًا في مختبره الشهير في Menlo Park."

    system_message = (
        "You are a professional multilingual NLP data parser.\n"
        "Split the provided text into meaningful, self-contained semantic chunks.\n"
        "Preserve the original text exactly and respect its language."
    )

    schema_str = '{"properties": {"original_text_length": {"type": "integer"}, "semantic_chunks": {"items": {"type": "string"}, "minItems": 2, "type": "array"}}, "required": ["original_text_length", "semantic_chunks"], "type": "object"}'

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"# Input Text:\n{input_text}\n\n# Output Schema:\n{schema_str}\n\n# Output JSON:\n```json\n"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    response = requests.post("http://localhost:8000/v1/completions", json={
        "model": "arabic-chunker",
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.1
    })

    print(response.json()["choices"][0]["text"])

**Output:**
```json
{
  "original_text_length": 102,
  "semantic_chunks": [
    "أسلوب إديسون في العمل كان يعتمد على Trial and Error.",
    "كان يمتلك فريقًا في مختبره الشهير في Menlo Park."
  ]
}
```

---

## Performance & Load Testing

Stress testing was conducted using **Locust** on a Google Colab T4 GPU (15 GB VRAM)
running **vLLM** with a production-stable configuration
(`--max-model-len 2048`, `--gpu-memory-utilization 0.8`).

### Load Test Results

| Metric | Result |
| :--- | :--- |
| **Concurrent Users** | 20 (simulated via Locust) |
| **Test Duration** | 60 seconds |
| **Total Requests** | 40 |
| **Failure Rate** | **0.00%** |
| **Throughput** | **437.20 tokens/sec** |
| **GPU KV Cache Usage** | ~3% (significant headroom remaining) |

### Response Time Percentiles

| Percentile | Latency |
| :--- | :--- |
| 50th (Median) | 19.0 s |
| 75th | 22.0 s |
| 90th | 24.0 s |
| 100th (Max) | 26.0 s |

*High per-request latency reflects the input text length used during testing (400–600 Arabic characters per request). vLLM's continuous batching kept GPU utilization stable throughout the test with zero failures across all 20 concurrent users.*

---

<details>
<summary> <b>Engineering Insights & Production Analysis (Click to expand)</b></summary>

### 1. Schema Adherence & Granularity Evolution
* **Strict Format Enforcement:** The pre-trained Baseline model failed to respect the structured JSON schema, returning an invalid layout (nested dictionaries `[{"text": "..."}]`) instead of raw strings. Post-SFT, the fine-tuned model achieved **100% schema adherence**, outputting perfectly structured JSON matching the exact required Pydantic schema.
* **Granularity Control:** The Teacher model (Gemini) chunked text at a coarse paragraph level (generating fewer, larger blocks). Interestingly, the fine-tuned model adapted a much finer, sentence-level granularity. For **Advanced RAG pipelines**, this granular approach is highly optimized as it avoids context dilution and significantly enhances dense vector embedding retrieval matching.

### 2. Dataset Distribution & Overfitting Analysis
* **Dataset Composition:** Built a distilled dataset of **2,257 total examples** mixed intentionally: ~65% News/Editorial (XLSum), ~25% Encyclopedic (Wikipedia), and ~10% Technical Mixed-Language (Code-Switching) to sustain strong cross-lingual performance.
* **The Validation Curve Lesson:** Evaluation showed that the lowest Validation Loss was achieved at **Step 800 (Val Loss: 0.867)**. Beyond Step 800, the Training Loss continued to decrease while the Validation Loss stagnated and drifted slightly, signaling classic early-stage overfitting.
* **Production Deployment Trade-off:** Since checkpoints were saved every 500 steps, `checkpoint-1000` was selected for production serving. The variance in validation loss between step 800 and 1000 was marginal (~0.008), meaning a complete re-train was economically inefficient under free-tier VRAM constraints.

### 3. Mitigating Qwen Chinese Character Leakage
* **The Issue:** Base Qwen architectures occasionally leak Chinese Unicode tokens during text generation due to native tokenizer vocabulary and pre-training dataset biases.
* **The Solution:** For standard `transformers` inference pipelines, an active **Logits Processor** was developed to dynamically apply a restrictive mask (`-inf`) over the Chinese Unicode character range (`0x4E00`-`0x9FFF`) during token sampling.
* **Serving Note:** When deployed via **vLLM** at a low `temperature=0.1` combined with a rigid system prompt, character leakage naturally dropped to **0.00%**, ensuring pristine Arabic and English output text streams without extra runtime parsing overhead.

</details>

---

## License
This project is licensed under the Apache License 2.0.
