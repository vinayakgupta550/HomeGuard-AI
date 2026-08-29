import time
import json
import os
import sys
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Add src/ directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from agent import SecurityAgent

def run_evaluation(num_samples: int = 10):
    print("=" * 60)
    print("🚀 HomeGuard-AI: Hugging Face Benchmark Evaluation Suite")
    print("=" * 60)

    print("\n[1/4] Loading Moondream2 VLM on Apple Silicon (MPS)...")
    model_id = "vikhyatk/moondream2"
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("mps")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    agent = SecurityAgent()

    print(f"\n[2/4] Streaming {num_samples} sample images from Hugging Face...")
    dataset = load_dataset("detection-datasets/coco", split="train", streaming=True)

    y_true, y_pred, latencies, results = [], [], [], []
    prompt = "Describe the main subject in this image. Is there a person, weapon, smoke, or everyday object?"

    # COCO ID Mappings for Ground Truth Threat Levels
    ALARM_CLASSES = [39, 43, 87]  # Baseball bats, knives, scissors (approximating weapons)
    PERSON_CLASSES = [0, 1]       # COCO person IDs

    print("\n[3/4] Running inference & tool evaluation...")
    count = 0
    
    for sample in dataset:
        if count >= num_samples:
            break

        image = sample["image"]
        
        # --- THE FIX: Parse Hugging Face's Dictionary of Lists ---
        objects = sample.get("objects", {})
        categories = objects.get("category", []) if isinstance(objects, dict) else []

        # 3-Tier Ground Truth Logic
        if any(c in ALARM_CLASSES for c in categories):
            ground_truth = "alarm"
        elif any(c in PERSON_CLASSES for c in categories):
            ground_truth = "alert"
        else:
            ground_truth = "log"

        # Run Inference
        start_t = time.perf_counter()
        enc_image = model.encode_image(image)
        observation = model.answer_question(enc_image, prompt, tokenizer)
        decision = agent.evaluate_and_act(observation)
        end_t = time.perf_counter()

        elapsed_sec = end_t - start_t
        latencies.append(elapsed_sec)
        
        pred_action = decision["action_taken"]
        y_true.append(ground_truth)
        y_pred.append(pred_action)

        print(f"Sample {count+1:03d}/{num_samples:03d} | Latency: {elapsed_sec*1000:6.1f} ms | Truth: {ground_truth:<5} | Pred: {pred_action:<5} | {'✅' if ground_truth == pred_action else '❌'}")

        results.append({
            "sample_id": count + 1,
            "latency_ms": round(elapsed_sec * 1000, 2),
            "ground_truth": ground_truth,
            "predicted_action": pred_action,
            "observation": observation,
            "match": ground_truth == pred_action
        })
        count += 1

    # Calculate and Display Metrics
    print("\n[4/4] Computing Quantitative Metrics...")
    accuracy = accuracy_score(y_true, y_pred) * 100
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    fps = 1.0 / (sum(latencies) / len(latencies))

    print("\n" + "=" * 60)
    print("📊 BENCHMARK SUMMARY REPORT (Apple Silicon)")
    print("=" * 60)
    print(f"• Total Evaluated Samples : {num_samples}")
    print(f"• Tool Selection Accuracy : {accuracy:.2f}%")
    print(f"• Mean Inference Latency  : {avg_latency_ms:.2f} ms")
    print(f"• Edge Throughput         : {fps:.2f} FPS")
    print("=" * 60)

    # Save to disk
    os.makedirs("data", exist_ok=True)
    with open("data/benchmark_results.json", "w") as f:
        json.dump({"metrics": {"accuracy_percent": round(accuracy, 2), "avg_latency_ms": round(avg_latency_ms, 2), "fps": round(fps, 2)}, "samples": results}, f, indent=2)

if __name__ == "__main__":
    run_evaluation(num_samples=10)