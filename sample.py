import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
import argparse

# =============================================================================
#
# LLaVA BENCHMARK RUNNER SCRIPT
#
# This script:
# 1. Loads a 'benchmark_data.json' file (created by generate_benchmark.py).
# 2. Loads the local LLaVA model.
# 3. Runs the benchmark on the data from the JSON file.
# 4. Prints the final accuracy.
#
# =============================================================================

# -------------------------------
# CONFIGURATION
# -------------------------------
# 1. Point this to your local LLaVA model
BASE_MODEL_PATH = "/home/naveenkumar/load/llava-model-local" 

# 2. Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def load_benchmark_image(image_path, base_dir):
    """
    Load a PIL Image from a file path for the benchmark.
    Combines the base_dir with the relative image_path.
    """
    # Create the full, absolute path
    # os.path.expanduser handles the '~' (home directory) symbol
    full_path = os.path.expanduser(os.path.join(base_dir, image_path))

    if not os.path.exists(full_path):
        print(f"Error: Image file not found at {full_path}")
        return None
    try:
        # Open and convert to RGB (LLaVA processor expects RGB)
        return Image.open(full_path).convert("RGB")
    except Exception as e:
        print(f"Error reading image {full_path}: {e}")
        return None

def run_kiva_benchmark(benchmark_data, model, processor, image_base_dir):
    """
    Runs the full benchmark loop on the provided data.
    """
    print(f"\n--- Running KiVA Benchmark on HF model: {BASE_MODEL_PATH} ---")

    correct = 0
    total = len(benchmark_data)

    if total == 0:
        print("Benchmark dataset is empty. No data found in JSON file.")
        return

    for idx, item in enumerate(benchmark_data, 1):
        print(f"\n--- Test Item {idx}/{total} ---")

        # Load all three images as PIL objects, passing the base directory
        img_input = load_benchmark_image(item["input_image"], image_base_dir)
        img_option_a = load_benchmark_image(item["option_image_a"], image_base_dir)
        img_option_b = load_benchmark_image(item["option_image_b"], image_base_dir)

        if not img_input or not img_option_a or not img_option_b:
            print(f"Skipping item {idx} due to one or more missing images.")
            # Print the original relative paths for clarity
            print(f"  Missing Input: {item['input_image']}")
            print(f"  Missing Option A: {item['option_image_a']}")
            print(f"  Missing Option B: {item['option_image_b']}")
            continue

        try:
            # Build the prompt
            prompt_content = f"""USER: <image>\n<image>\n<image>\nThe first image is the input. The second image is Option A. The third image is Option B.

Question: {item['question']}

Based on the input image, which option (A or B) is the correct answer?
Please respond with only the letter 'A' or 'B'.
ASSISTANT:"""

            # Process images and text
            inputs = processor(
                text=prompt_content,
                images=[img_input, img_option_a, img_option_b],
                return_tensors="pt"
            ).to(DEVICE)

            # Generate response
            with torch.inference_mode():
                output_ids = model.generate(**inputs, max_new_tokens=10)

            # Decode the output
            model_answer_raw = processor.batch_decode(
                output_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            # Clean the output
            if "ASSISTANT:" in model_answer_raw:
                 model_answer_raw = model_answer_raw.split("ASSISTANT:")[-1].strip()

            ground_truth = item["ground_truth_answer"].upper()

            print(f"Question: {item['question']}")
            print(f"Ground Truth: {ground_truth}")
            print(f"Model Answer (Raw): {model_answer_raw}")

            # Evaluation logic
            model_choice = ""
            for char in model_answer_raw.upper():
                if char in ('A', 'B'):
                    model_choice = char
                    break  # Found the first choice

            if model_choice == ground_truth:
                print(f"Result: CORRECT ✅ (Model chose {model_choice})")
                correct += 1
            else:
                print(f"Result: INCORRECT ❌ (Expected: {ground_truth}, Got: {model_choice or 'None'})")

        except Exception as e:
            print(f"Error processing item {idx}: {e}")

    # Final results
    if total > 0:
        accuracy = (correct / total) * 100
        print("\n--- Benchmark Complete ---")
        print(f"Total Items: {total}")
        print(f"Correct Predictions: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("\nNo valid items processed.")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run LLaVA benchmark from a JSON file.")
    # This argument is required to tell the script where to find the JSON file
    parser.add_argument('--benchmark_file', type=str, required=True,
                        help='Path to the benchmark_data.json file (created by generate_benchmark.py).')
    # This new argument tells the script where the image folders are located
    parser.add_argument('--image_base_dir', type=str, required=True,
                        help='The absolute base path to the directory containing the images (e.g., ~/fyp).')
    args = parser.parse_args()

    # --- Part 1: Load Benchmark Data ---
    print("--- PART 1: LOADING BENCHMARK DATA ---")
    
    # --- THIS IS THE FIX ---
    # Expand the user path (handles '~') for the benchmark file
    benchmark_file_path = os.path.expanduser(args.benchmark_file)
    # -----------------------

    try:
        # Use the expanded path
        with open(benchmark_file_path, 'r') as f:
            benchmark_data_list = json.load(f)
        print(f"Loaded {len(benchmark_data_list)} items from {benchmark_file_path}")
    except Exception as e:
        # Print the path that was tried
        print(f"CRITICAL ERROR: Could not load benchmark file from {benchmark_file_path}.")
        print(f"Error: {e}")
        print("Please run 'generate_benchmark.py' first and check the path.")
        return

    # --- Part 2: Load Model ---
    print("\n--- PART 2: LOADING LLaVA MODEL ---")
    print(f"Loading base LLaVA model from {BASE_MODEL_PATH}...")
    try:
        processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
        model = LlavaForConditionalGeneration.from_pretrained(
            BASE_MODEL_PATH, 
            # --- THIS IS THE FIX for the warning ---
            # Changed 'torch_dtype' to 'dtype'
            dtype=torch.float16 if DEVICE=="cuda" else torch.float32
            # -------------------------------------
        ).to(DEVICE)
        print(f"Model loaded to {DEVICE} ✅")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model from {BASE_MODEL_PATH}.")
        print(f"Error: {e}")
        return

    # --- Part 3: Run Benchmark ---
    # Pass the data we loaded and the new image_base_dir
    run_kiva_benchmark(benchmark_data_list, model, processor, args.image_base_dir)

if __name__ == "__main__":
    main()

