import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# -------------------------------
# CONFIGURATION
# -------------------------------
# Use the same model ID you used before
BASE_MODEL_PATH = "/home/naveenkumar/load/llava-model-local"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 1️⃣ Load Base Model
# -------------------------------
print(f"Loading base LLaVA model from {BASE_MODEL_PATH}...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
model = LlavaForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH, 
    torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32
).to(DEVICE)
print("Model loaded ✅")

# -------------------------------
# 2️⃣ Ask a Question (Text-Only)
# -------------------------------
print("\nAsking a simple question...")

question = "What is the capital of India?"

# Format the prompt for a text-only chat
# We remove the <image> token
prompt = f"USER: {question} ASSISTANT:"

# Process *only* the text
inputs = processor(
    text=prompt, 
    images=None,  # Explicitly pass no images
    return_tensors="pt"
).to(DEVICE)

# -------------------------------
# 3️⃣ Get the Answer
# -------------------------------
# Generate the response
# The model will run in text-only mode since pixel_values is None
output = model.generate(**inputs, max_new_tokens=50)

# Decode the output
full_output = processor.batch_decode(output, skip_special_tokens=True)[0]
print(f"\nFull model output: {full_output}")

# Extract just the answer part
try:
    predicted_answer = full_output.split("ASSISTANT:")[1].strip()
    print(f"\nPredicted Answer: {predicted_answer}")
except IndexError:
    print("Error: Could not parse the model's answer.")