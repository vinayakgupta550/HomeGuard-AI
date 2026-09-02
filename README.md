# HomeGuard-AI: Edge Vision-Language Agent for Autonomous Threat Detection

HomeGuard-AI is an edge-native, real-time security monitoring pipeline designed for Apple Silicon. It pairs small Vision-Language Models (sVLMs) with an autonomous agentic decision loop to achieve local frame ingestion, semantic scene reasoning, and automated tool dispatching without sending raw video feeds off-device.

---

## 🏗️ System Architecture

The pipeline implements an agentic **Sense $\rightarrow$ Think $\rightarrow$ Act** control loop:

```
[ Camera Feed / Test Dataset ]
               │
               ▼ (Sense)
   [ OpenCV Sensor Ingestion ]
   └── Preprocessing & Fixed Tensor Resizing (512x512)
               │
               ▼ (Think - Perception)
  [ MLX-VLM Engine (Apple Silicon) ]
   ├── 4-Bit Quantized sVLM (Qwen2-VL / Moondream)
   ├── Automatic Prefix Caching (KV-Cache Reuse)
   └── Structured JSON Output Constraints
               │
               ▼ (Act - Decision & Dispatch)
        [ SecurityAgent ]
   ├── Threat Evaluation Heuristics
   └── Tool Execution Registry
         ├── trigger_alarm()  [Critical Threat: Weapons/Smoke]
         ├── notify_user()    [Warning: Unauthorized Human Presence]
         └── log_incident()   [Normal: Routine Logging]

```

---

## ⚡ Key Optimizations on Apple Silicon

1. **Native MLX Unified Memory Utilization:** Replaced PyTorch MPS execution with Apple MLX (`mlx-vlm`), enabling zero-copy memory transfers between CPU and GPU.
2. **4-Bit Weight Quantization:** Reduces memory footprint to under 2GB VRAM while doubling token-per-second memory bandwidth efficiency.
3. **Automatic Prefix Caching (KV-Cache Reuse):** Enables `APC_ENABLED="1"` to avoid re-encoding system prompts and tool definitions on subsequent iterations.
4. **Constrained JSON Decoding:** Forces structured token output, bounding generation to fewer than 40 tokens per cycle and eliminating conversational overhead.
5. **Fixed Tensor Ingestion:** Normalizes frames to $512\times 512$ before inference, preventing repeated MLX JIT graph recompilations.

---

## 📊 Empirical Evaluation & Benchmarks

The project includes an automated evaluation harness (`tests/evaluate.py`) streaming labeled benchmark data directly from Hugging Face (`detection-datasets/coco`).

### Optimization Impact

| Configuration | Framework | Model Precision | Mean Latency | Throughput |
| --- | --- | --- | --- | --- |
| Baseline | PyTorch (`mps`) | FP16 | ~2,268 ms | ~0.44 FPS |
| Optimized | Apple MLX (`mlx-vlm`) | 4-bit Quantized | **~748 ms** | **~1.34 FPS** |

*Hardware: Apple Silicon M-Series Unified Memory Architecture.*

---

## 📁 Repository Structure

```
HomeGuard-AI/
├── data/                  # Ignored from Git: test images, logs, benchmarks
├── src/
│   ├── capture.py         # Hardware camera sensor ingestion & warm-up
│   ├── agent.py           # SecurityAgent class and tool dispatch registry
│   ├── vlm_engine.py      # PyTorch baseline VLM implementation
│   └── main.py            # Optimized MLX orchestrator pipeline
├── tests/
│   └── evaluate.py        # Hugging Face benchmark & metrics evaluation harness
├── requirements.txt       # Pinned project dependencies
├── .gitignore             # Excludes large binaries, model weights, and local data
└── README.md              # Project architecture and technical documentation

```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/HomeGuard-AI.git
cd HomeGuard-AI

# Create and activate Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Run the Live Edge Pipeline

```bash
python src/main.py

```

### 3. Run Benchmark Evaluations

```bash
python tests/evaluate.py

```

---

## 🛠️ Tech Stack

* **Core Framework:** Apple MLX / `mlx-vlm`
* **Models Tested:** `mlx-community/Qwen2-VL-2B-Instruct-4bit`, `vikhyatk/moondream2`, `mlx-community/SmolVLM-Instruct-4bit`, `mlx-community/Phi-3.5-vision-instruct-4bit`, `mlx-community/Qwen2-VL-7B-Instruct-4bit`
* **Data & Benchmarking:** Hugging Face `datasets`, `scikit-learn`, `pandas`
* **Perception Ingestion:** `OpenCV`, `Pillow`

## v1.0 Prototyping: Naive VLM Pipeline

v1.0 explored using the following models via MLX:

1. Qwen2-VL-2B-Instruct-4bit
2. SmolVLM-Instruct-4bit
3. Phi-3.5-vision-instruct-4bit
4. Qwen2-VL-7B-Instruct-4bit

for direct object detection on a COCO subset. While the model achieved high accuracy through prompt engineering and regex-based output parsing (to bypass JSON/bounding box hallucinations), the architecture revealed two major flaws for edge deployment: it bottlenecked at ~1.2 FPS, and VLMs struggle with static object counting. This led to the architectural pivot in v2.0.