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
import re

# Import the local agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from agent import SecurityAgent

os.environ["APC_ENABLED"] = "1"
FIXED_SIZE = (512, 512)

def process_image(img, filepath="data/eval_frame.jpg"):
    """Resizes to a fixed tensor shape to leverage MLX JIT caching."""
    img = img.convert("RGB").resize(FIXED_SIZE)
    img.save(filepath)
    return filepath

def evaluate_model(model_path: str, num_samples: int = 50):
    print(f"\n" + "="*60)
    print(f"🧪 EVALUATING: {model_path}")
    print("="*60)

    # 1. Load Model
    model, processor = load(model_path)
    config = load_config(model_path)
    agent = SecurityAgent()

    # Determine agent processing method (compatibility check)
    agent_func = agent.evaluate_and_act if hasattr(agent, "evaluate_and_act") else agent.process_observation

    prompt = """Analyze the image and describe it in one short sentence.

    Follow these rules in order:
    IF you see a weapon, fire, smoke, or mask -> Output ACTION: LEVEL_1 (Highest priority)
    ELSE IF you see a person, people, man, woman, child, rider, or vehicle -> Output ACTION: LEVEL_2
    ELSE you see NONE of the above -> Output ACTION: LEVEL_3 (Lowest priority)

    DO NOT output bounding boxes. Format exactly like this:
    Observation: [your description]
    Action: [LEVEL_1, LEVEL_2, or LEVEL_3]
    """
    
    # MLX dynamically handles formatting for Qwen, SmolVLM, and Llama!
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)

    # 2. Warm-up Neural Engine
    print("Warming up MLX JIT graph...")
    dummy_img = Image.new('RGB', FIXED_SIZE, color='black')
    dummy_path = process_image(dummy_img, "data/dummy.jpg")
    _ = generate(model, processor, formatted_prompt, [dummy_path], max_tokens=10, verbose=False)

    # 3. Stream Data
    dataset = load_dataset("detection-datasets/coco", split="train", streaming=True, token=True)
    # 3. Cache Data Locally (Bypasses network throttling)
    # print("Fetching dataset to local cache...")
    # dataset = load_dataset("detection-datasets/coco", split="val[:50]")
    ALARM_CLASSES = [39, 43, 87]
    PERSON_CLASSES = [0, 1]
    VEHICLE_CLASSES = [3, 4, 6, 8] # COCO IDs for car, motorcycle, bus, truck

    y_true, y_pred, latencies = [], [], []
    count = 0
    # Collect errors for failure mode analysis
    errors = []

    print("Running Inference...")
    for idx, sample in enumerate(dataset):
        if count >= num_samples: break
        
        img_path = process_image(sample["image"])
        objects = sample.get("objects", {})
        categories = objects.get("category", []) if isinstance(objects, dict) else []
        
        if any(c in ALARM_CLASSES for c in categories): gt_action = "alarm"
        elif any(c in PERSON_CLASSES for c in categories) or any(c in VEHICLE_CLASSES for c in categories): 
            gt_action = "alert"
        else: gt_action = "log"

        # Timer starts
        start_t = time.perf_counter()
        # Extract just the text property and give it 150 tokens to finish the JSON
        # Generate the response
        # Give it 50 tokens to write its observation and action
        result_obj = generate(model, processor, formatted_prompt, [img_path], max_tokens=150, verbose=False)
        raw_text = str(getattr(result_obj, "text", result_obj)).strip().upper()
        
        # Strip out any spatial grounding tags like <|BOX_START|>
        clean_text = re.sub(r'<\|.*?\|>', '', raw_text)
        
        # Target ONLY the exact Level
        match = re.search(r'ACTION:\s*(LEVEL_1|LEVEL_2|LEVEL_3)', clean_text)
        
        if match:
            level = match.group(1)
            if level == "LEVEL_1":
                pred_action = "alarm"
            elif level == "LEVEL_2":
                pred_action = "alert"
            elif level == "LEVEL_3":
                pred_action = "log"
        else:
            # Fallback if it forgot the "Action:" prefix
            if "LEVEL_1" in clean_text:
                pred_action = "alarm"
            elif "LEVEL_2" in clean_text:
                pred_action = "alert"
            else:
                pred_action = "log"
                
        # Stop the timer
        end_t = time.perf_counter()
        elapsed_ms = (end_t - start_t) * 1000
        
        # Normalize naming for accuracy calc
        if "log" in pred_action: pred_action = "log"
        if "alert" in pred_action or "notify" in pred_action: pred_action = "alert"
        if "alarm" in pred_action: pred_action = "alarm"

        latencies.append(elapsed_ms)
        y_true.append(gt_action)
        y_pred.append(pred_action)
        count += 1
        if pred_action != gt_action:
            # Save the misclassified image to your disk
            error_img_filename = f"error_sample_{idx}_gt_{gt_action}.jpg"
            sample["image"].save(error_img_filename)
            
            errors.append({
                "sample_idx": idx,
                "gt_action": gt_action,
                "predicted": pred_action,
                "raw_response": raw_text, # Now this is a clean string
                "categories_present": categories,
                "saved_image": error_img_filename
            })

    accuracy = accuracy_score(y_true, y_pred) * 100
    avg_latency = sum(latencies) / len(latencies)

    print(f"\n❌ MISCLASSIFICATIONS ({len(errors)}/{num_samples}):")
    for err in errors:
        print(f"  • Sample #{err['sample_idx']} | GT: '{err['gt_action']}' ➔ Model Predicted: '{err['predicted']}'")
        print(f"    COCO Classes Present: {err['categories_present']}")
        # Wrapping it in str() safely extracts the text representation
        print(f"    Raw Output: {str(err['raw_response']).strip()}\n")
    
    # 4. Critical Step: Destroy the model and free GPU memory before the next loop
    del model, processor, agent
    print_evaluation_metrics(y_true, y_pred, labels=["log", "alert", "alarm"])
    mx.clear_cache()
    
    return {
        "model": model_path.split("/")[-1], 
        "accuracy": round(accuracy, 2), 
        "latency_ms": round(avg_latency, 2), 
        "fps": round(1000 / avg_latency, 2)
    }

from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

def print_evaluation_metrics(y_true, y_pred, labels=["log", "alert", "alarm"]):
    print("\n" + "=" * 60)
    print("📊 DETAILED METRICS BREAKDOWN")
    print("=" * 60)
    
    # 1. Distribution Counts
    gt_counts = Counter(y_true)
    pred_counts = Counter(y_pred)
    
    print("\n[Class Distribution]")
    print(f"{'Class':<10} | {'Ground Truth':<15} | {'Predicted':<15}")
    print("-" * 45)
    for label in labels:
        print(f"{label:<10} | {gt_counts.get(label, 0):<15} | {pred_counts.get(label, 0):<15}")
        
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\n[Confusion Matrix]")
    header = "Pred ➔".ljust(10) + "".join([f"{l:>10}" for l in labels])
    print(header)
    print("-" * len(header))
    for idx, true_label in enumerate(labels):
        row_str = f"GT: {true_label}".ljust(10) + "".join([f"{cm[idx][j]:>10}" for j in range(len(labels))])
        print(row_str)

    # 3. Precision, Recall, F1-Score
    print("\n[Per-Class Performance Report]")
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    print(report)

if __name__ == "__main__":
    # The models to compare
    models_to_test = [
        "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        "mlx-community/SmolVLM-Instruct-4bit",
        # # Uncomment the line below if your Mac has 16GB+ of Unified Memory
        # # "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit" 
        "mlx-community/Phi-3.5-vision-instruct-4bit",
        "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    ]
    
    results = []
    # We use 50 samples for speed during the ablation loop
    for m in models_to_test:
        results.append(evaluate_model(m, num_samples=20))
        
    print("\n\n" + "="*80)
    print("🏆 HOMEGUARD-AI: ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Model Name':<45} | {'Acc (%)':<8} | {'Latency (ms)':<14} | {'FPS':<6}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<45} | {r['accuracy']:<8.2f} | {r['latency_ms']:<14.1f} | {r['fps']:<6.2f}")
    print("="*80)