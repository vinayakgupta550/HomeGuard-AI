import time
import json
import os
import sys
from PIL import Image
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from agent import SecurityAgent

os.environ["APC_ENABLED"] = "1"
FIXED_SIZE = (512, 512) # Prevents MLX from recompiling graphs!

def process_image_for_mlx(img, filepath="data/eval_frame.jpg"):
    """Resizes to a fixed tensor shape to leverage MLX JIT caching."""
    img = img.convert("RGB").resize(FIXED_SIZE)
    img.save(filepath)
    return filepath

def run_evaluation(num_samples: int = 100):
    print("=" * 60)
    print("🚀 HomeGuard-AI: MLX 4-bit Edge Benchmark")
    print("=" * 60)

    print("\n[1/4] Loading Qwen2-VL 4-bit into unified memory...")
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    model, processor = load(model_path)
    config = load_config(model_path)
    agent = SecurityAgent()

    prompt = """Analyze the image and respond ONLY with a valid JSON object. Do not add explanation.
    Schema: { "observation": "Brief 1-sentence description", "detected_threat": "person" | "weapon" | "smoke" | "none" }"""
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)

    # --- WARM-UP ---
    print("\n[Warming up Neural Engine for 512x512 inputs...]")
    dummy_img = Image.new('RGB', FIXED_SIZE, color='black')
    dummy_path = process_image_for_mlx(dummy_img, "data/dummy.jpg")
    _ = generate(model, processor, formatted_prompt, [dummy_path], max_tokens=10, verbose=False)
    print("Graph compiled! Real-time latency unlocked.")

    print(f"\n[2/4] Streaming {num_samples} samples from Hugging Face...")
    dataset = load_dataset("detection-datasets/coco", split="train", streaming=True)

    ALARM_CLASSES = [39, 43, 87]  # Weapons/Sharp objects
    PERSON_CLASSES = [0, 1]       # Humans

    y_true, y_pred, latencies, results = [], [], [], []
    count = 0

    print("\n[3/4] Running High-Speed Inference...")
    for sample in dataset:
        if count >= num_samples: break
        
        # Standardize image size
        img_path = process_image_for_mlx(sample["image"])

        # Determine ground truth (matching your agent's tool names)
        objects = sample.get("objects", {})
        categories = objects.get("category", []) if isinstance(objects, dict) else []
        
        if any(c in ALARM_CLASSES for c in categories): gt_action = "alarm"
        elif any(c in PERSON_CLASSES for c in categories): gt_action = "alert"
        else: gt_action = "log"

        # INFERENCE
        start_t = time.perf_counter()
        raw_output = generate(model, processor, formatted_prompt, [img_path], max_tokens=40, verbose=False)
        
        # Parse output safely
        raw_text = getattr(raw_output, "text", str(raw_output))
        try:
            clean_output = raw_text.replace("```json", "").replace("```", "").strip()
            obs = json.loads(clean_output).get("observation", "Unknown")
        except json.JSONDecodeError:
            obs = raw_text
            
        decision = agent.evaluate_and_act(obs) if hasattr(agent, "evaluate_and_act") else agent.process_observation(obs)
        end_t = time.perf_counter()

        # Record metrics
        elapsed_ms = (end_t - start_t) * 1000
        pred_action = decision.get("action_taken", decision.get("action", "unknown"))
        
        latencies.append(elapsed_ms)
        y_true.append(gt_action)
        y_pred.append(pred_action)

        match_icon = '✅' if gt_action == pred_action else '❌'
        print(f"Sample {count+1:03d} | Latency: {elapsed_ms:6.1f} ms | Truth: {gt_action:<13} | Pred: {pred_action:<13} | {match_icon}")
        
        results.append({"sample_id": count+1, "latency_ms": round(elapsed_ms, 2), "truth": gt_action, "pred": pred_action})
        count += 1

    # SUMMARY
    print("\n[4/4] Computing Apple Silicon Metrics...")
    accuracy = accuracy_score(y_true, y_pred) * 100
    avg_latency = sum(latencies) / len(latencies)
    fps = 1000 / avg_latency

    print("\n" + "=" * 60)
    print("📊 FINAL BENCHMARK SUMMARY (MLX 4-bit + KV Cache)")
    print("=" * 60)
    print(f"• Evaluated Samples       : {num_samples}")
    print(f"• Tool Selection Accuracy : {accuracy:.2f}%")
    print(f"• Mean Inference Latency  : {avg_latency:.2f} ms")
    print(f"• Edge Throughput         : {fps:.2f} FPS")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)
    with open("data/benchmark_results.json", "w") as f:
        json.dump({"accuracy": accuracy, "avg_latency_ms": avg_latency, "fps": fps}, f, indent=2)

if __name__ == "__main__":
    run_evaluation(num_samples=10)