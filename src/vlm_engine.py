import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time

print("Loading Moondream2 VLM into Apple Silicon unified memory...")
print("(This will take a minute or two on the first run as it downloads the model)")

model_id = "vikhyatk/moondream2"

# 1. Load the model and tokenizer, explicitly sending them to the Mac's Metal Performance Shaders (MPS)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True
).to("mps")

tokenizer = AutoTokenizer.from_pretrained(model_id)
print("Model loaded successfully!")

# 2. Load the hardware frame we captured in Step 7
image = Image.open("data/test_frame.jpg")

# 3. Process the image and ask the AI a question
print("\nAnalyzing the frame...")
start_time = time.time()

# The model translates the image into math (embeddings)
enc_image = model.encode_image(image)
# The model reasons about the question using the image math
answer = model.answer_question(enc_image, "Describe the main subject in this image.", tokenizer)

end_time = time.time()

print(f"\nAI Observation: {answer}")
print(f"Edge Latency: {end_time - start_time:.2f} seconds")