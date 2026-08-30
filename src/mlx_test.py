from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import time

print("Loading 4-bit quantized MLX VLM into Apple Silicon unified memory...")
print("(First run will take a moment to download the optimized 4-bit weights)")

# 1. Load the pre-quantized model. 
model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
model, processor = load(model_path)
config = load_config(model_path)

# 2. Define the inputs
image_path = "data/test_frame.jpg"
prompt = "Describe the main subject in this image. Keep it brief."

# 3. Format the prompt using the model's chat template
formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=1
)

# 4. Run Inference and time it
print("\nAnalyzing the frame with MLX...")
start_t = time.perf_counter()

output = generate(model, processor, formatted_prompt, [image_path], verbose=False)

end_t = time.perf_counter()

print(f"\nAI Observation: {output}")
print(f"⚡️ MLX 4-bit Latency: {(end_t - start_t) * 1000:.1f} ms")