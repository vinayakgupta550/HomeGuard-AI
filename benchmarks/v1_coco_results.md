# Phase 1: Edge VLM Benchmark & Ablation Study (COCO Subset)

## 1. Executive Summary

This study benchmarks edge-deployable Vision-Language Models (VLMs) running natively on Apple Silicon using the MLX framework (`mlx-vlm`). The objective was to assess the viability of standalone 4-bit quantized VLMs for real-time security surveillance and event classification (LOG vs. ALERT vs. ALARM) directly on edge hardware.

---

## 2. Hardware & Runtime Configuration

- **Platform:** Apple Silicon (Unified Memory Architecture)
- **Framework:** `mlx-vlm` (4-bit quantization, MLX JIT execution)
- **Dataset:** COCO 2017 Validation Subset ($N = 20$ balanced evaluation frames)
- **Task:** Zero-shot scene classification into operational tiers:
  - `LOG`: Safe background, nature, animals, static environment.
  - `ALERT`: Presence of humans (pedestrians, riders) or vehicles.
  - `ALARM`: Direct threats (weapons, smoke/fire, concealed faces).

---

## 3. Benchmark Results

| Model Architecture | Parameter Count | Quantization | Accuracy (%) | Latency (ms) | Throughput (FPS) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2-VL-2B-Instruct** | 2.2B | 4-bit | 50.00% | **868.4** | **1.15** |
| **SmolVLM-Instruct** | 2.2B | 4-bit | 50.00% | 2530.1 | 0.40 |
| **Phi-3.5-vision-instruct** | 4.2B | 4-bit | 60.00% | 1534.6 | 0.65 |
| **Qwen2-VL-7B-Instruct** | 7.6B | 4-bit | **75.00%** | 1778.2 | 0.56 |

---

## 4. Detailed Per-Class Breakdown

### Qwen2-VL-2B-Instruct-4bit
- **Overall Accuracy:** 50.00% | **Macro Avg F1:** 0.22
- **Confusion Matrix:**
  - `GT: log` (10 samples) $\rightarrow$ 10 predicted `log`, 0 `alert`, 0 `alarm`
  - `GT: alert` (9 samples) $\rightarrow$ 9 predicted `log`, 0 `alert`, 0 `alarm`
  - `GT: alarm` (1 sample) $\rightarrow$ 1 predicted `log`, 0 `alert`, 0 `alarm`

### SmolVLM-Instruct-4bit
- **Overall Accuracy:** 50.00% | **Macro Avg F1:** 0.22
- **Confusion Matrix:**
  - `GT: log` (10 samples) $\rightarrow$ 10 predicted `log`, 0 `alert`, 0 `alarm`
  - `GT: alert` (9 samples) $\rightarrow$ 9 predicted `log`, 0 `alert`, 0 `alarm`
  - `GT: alarm` (1 sample) $\rightarrow$ 1 predicted `log`, 0 `alert`, 0 `alarm`

### Phi-3.5-vision-instruct-4bit
- **Overall Accuracy:** 60.00% | **Macro Avg F1:** 0.41
- **Confusion Matrix:**
  - `GT: log` (10 samples) $\rightarrow$ 6 predicted `log`, 4 `alert`, 0 `alarm`
  - `GT: alert` (9 samples) $\rightarrow$ 3 predicted `log`, 6 `alert`, 0 `alarm`
  - `GT: alarm` (1 sample) $\rightarrow$ 0 predicted `log`, 1 `alert`, 0 `alarm`

### Qwen2-VL-7B-Instruct-4bit
- **Overall Accuracy:** 75.00% | **Macro Avg F1:** 0.50
- **Confusion Matrix:**
  - `GT: log` (10 samples) $\rightarrow$ 10 predicted `log`, 0 `alert`, 0 `alarm`
  - `GT: alert` (9 samples) $\rightarrow$ 4 predicted `log`, 5 `alert`, 0 `alarm`
  - `GT: alarm` (1 sample) $\rightarrow$ 1 predicted `log`, 0 `alert`, 0 `alarm`

---

## 5. Key Technical Findings & Failure Modes

1. **Throughput Bottleneck:**
   The fastest model (`Qwen2-VL-2B`) peaked at **1.15 FPS** (~868 ms/frame). This throughput cannot support real-time 30 FPS RTSP ingestion without introducing massive frame drops and pipeline lag.
2. **Sub-4B Semantic Rebellion & False Negatives:**
   Models under 3B parameters (`Qwen2-VL-2B` and `SmolVLM`) exhibited complete recall collapse on the `ALERT` class (0% recall). Despite correctly recognizing entities in their textual descriptions (e.g., *"a woman holding a pink umbrella"*, *"two people riding horses"*), their safety alignment biased them toward classifying non-hostile human presence as safe (`LEVEL_3` / `LOG`).
3. **Contextual Hallucinations in Household Scenes:**
   Sample #17 (a domestic kitchen containing cooking knives) was classified as `LOG` across 2B and 7B models, illustrating the difficulty zero-shot VLMs face when distinguishing functional everyday tools from security threats without behavioral context.
4. **Scale vs. Latency Trade-off:**
   `Qwen2-VL-7B` achieved the highest accuracy (75.00%), successfully resolving ambiguous human and vehicle scenes. However, its 1.78s latency per frame makes standalone edge deployment impractical for live alerting.

---

## 6. Architectural Decision: Transition to v2.0

Standalone VLM inference on raw edge video feeds is structurally inefficient:
- Continuous forward passes saturate unified memory and thermal headroom.
- Generative language models are poorly suited for deterministic spatial object detection.

**v2.0 Strategy:** Decouple spatial detection from contextual reasoning via a **Cascading Hybrid Architecture**:
1. **L1 Watchdog:** Ultra-lightweight object detector (YOLOv11 via CoreML/MPS) running continuous inference at 30–60 FPS.
2. **L2 Cognitive Trigger:** Event-driven handoff to `Qwen2-VL-2B` only when human or vehicle presence is confirmed.
3. **Behavioral Evaluation:** Migrate benchmark dataset from COCO (static objects) to **UCF-Crime** (temporal behavioral anomalies) to evaluate intent rather than object presence.