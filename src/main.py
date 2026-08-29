import cv2
import time
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the professional modular agent we built in the previous step
from agent import SecurityAgent 

def capture_live_frame(filepath="data/live_frame.jpg"):
    """Handles edge hardware ingestion."""
    cap = cv2.VideoCapture(0)
    print("Warming up camera sensor...")
    time.sleep(2)
    
    for _ in range(10): # Flush buffer
        cap.read()
        
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filepath, frame)
    cap.release()
    return filepath if ret else None

def main():
    print("\n=== Initializing HomeGuard-AI Edge Pipeline ===")
    
    # 1. Boot up the Agent
    agent = SecurityAgent()
    
    # 2. Load the VLM into Apple Silicon (MPS)
    print("Loading VLM into unified memory...")
    model_id = "vikhyatk/moondream2"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True
    ).to("mps")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 3. Pipeline Step 1: SENSE
    print("\n[1/3] Capturing live environment...")
    img_path = capture_live_frame()
    if not img_path:
        print("Camera error. Exiting.")
        return
        
    # 4. Pipeline Step 2: THINK (Perception)
    print("[2/3] AI analyzing scene...")
    image = Image.open(img_path)
    enc_image = model.encode_image(image)
    
    # We ask a targeted security question to guide the VLM
    prompt = "Describe the main subject in this image. Is there a person, and what are they doing?"
    observation = model.answer_question(enc_image, prompt, tokenizer)
    print(f"      Observation: '{observation}'")
    
    # 5. Pipeline Step 3: ACT (Agent Decision)
    print("[3/3] Agent evaluating threat level...")
    result = agent.evaluate_and_act(observation)
    
    print("\n=== Execution Summary ===")
    print(f"Threat Status: {result['threat_level']}")
    print(f"Action Taken:  {result['tool_output']}")
    print("=========================================\n")

if __name__ == "__main__":
    main()