import cv2
import time
import json
import os
import numpy as np
from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from agent import SecurityAgent 

os.environ["APC_ENABLED"] = "1"

def capture_live_frame(filepath="data/live_frame.jpg"):
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    for _ in range(5): cap.read()
    ret, frame = cap.read()
    if ret: cv2.imwrite(filepath, frame)
    cap.release()
    return filepath if ret else None

def create_dummy_image(filepath="data/dummy.jpg"):
    """Creates a tiny black image to trigger MLX graph compilation."""
    os.makedirs("data", exist_ok=True)
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    img.save(filepath)
    return filepath

def main():
    print("\n=== Initializing MLX-Optimized Edge Pipeline ===")
    agent = SecurityAgent()
    
    print("Loading 4-bit Qwen2-VL into unified memory...")
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    model, processor = load(model_path)
    config = load_config(model_path)
    
    system_instruction = """
    Analyze the image and respond ONLY with a valid JSON object. Do not add explanation.
    Schema:
    {
      "observation": "Brief 1-sentence description",
      "detected_threat": "person" | "weapon" | "smoke" | "none"
    }"""
    
    formatted_prompt = apply_chat_template(processor, config, system_instruction, num_images=1)
    
    # --- WARM-UP JIT COMPILATION ---
    print("\n[Warming up Apple Silicon Neural Engine (JIT Compilation)...]")
    dummy_img = create_dummy_image()
    _ = generate(model, processor, formatted_prompt, [dummy_img], max_tokens=10, verbose=False)
    print("Warm-up complete. Hardware graph compiled.")
    
    # --- REAL PIPELINE ---
    print("\n[1/3] Capturing live environment...")
    img_path = capture_live_frame()
    
    print("[2/3] AI analyzing scene (KV Caching Active)...")
    start_t = time.perf_counter()
    
    raw_output = generate(model, processor, formatted_prompt, [img_path], max_tokens=40, verbose=False)
    
    # Extract text from GenerationResult safely
    raw_text = getattr(raw_output, "text", str(raw_output))
    
    try:
        clean_output = raw_text.replace("```json", "").replace("```", "").strip()
        decision_data = json.loads(clean_output)
        observation = decision_data.get("observation", "Unknown")
    except json.JSONDecodeError:
        observation = raw_text
        
    print(f"      Observation: '{observation}'")
    
    print("[3/3] Agent evaluating threat level...")
    result = agent.evaluate_and_act(observation)
    
    end_t = time.perf_counter()
    
    print("\n=== Execution Summary ===")
    # .get() safely checks for 'status', and falls back to 'threat_level' if using the older agent.py
    print(f"Threat Status: {result.get('status', result.get('threat_level', 'UNKNOWN'))}")
    print(f"Action Taken:  {result.get('tool_response', result.get('tool_output', 'UNKNOWN'))}")
    print(f"True Inference Latency: {(end_t - start_t) * 1000:.1f} ms")
    print("=========================================\n")

if __name__ == "__main__":
    main()