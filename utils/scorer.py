import os
import shutil
import sys
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

# Try to import the requested library
try:
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
except ImportError:
    print("Error: library 'aesthetic-predictor-v2-5' not found.")
    print("Please run: pip install aesthetic-predictor-v2-5")
    sys.exit(1)

# ================= CONFIGURATION =================
INPUT_FOLDER = r"."
OUTPUT_FOLDER = r"./best_images"
TOP_K = 50  # How many images to keep?

# Valid image extensions
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
# =================================================

# Handle truncated images (common in large datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup_device():
    """Determines valid device and data type."""
    if torch.cuda.is_available():
        device = "cuda"
        # BF16 is faster/efficient on modern GPUs (Ampere+), 
        # but older GPUs (Pascal) need Float16 or Float32.
        try:
            # fast check if bf16 is supported
            torch.zeros(1).to(device, dtype=torch.bfloat16)
            dtype = torch.bfloat16
            print("Device: CUDA (bfloat16)")
        except TypeError:
            dtype = torch.float16
            print("Device: CUDA (float16)")
    else:
        device = "cpu"
        dtype = torch.float32
        print("Device: CPU (float32) - Warning: This will be slow.")
    
    return device, dtype

def load_model(device, dtype):
    """
    Loads the aesthetic-predictor-v2-5 model (SigLIP based).
    """
    print("Loading aesthetic-predictor-v2-5...")
    
    # Load model and preprocessor
    model, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    model = model.to(dtype).to(device)
    model.eval() # Ensure eval mode
    
    return model, preprocessor

def get_image_score(image_path, model, preprocessor, device, dtype):
    """
    Opens an image and predicts its aesthetic score.
    """
    try:
        # Convert to RGB to handle PNGs with transparency or Grayscale
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"\nError opening {image_path}: {e}")
        return 0.0

    # Preprocess
    # The preprocessor returns a dict, we extract pixel_values
    inputs = preprocessor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(dtype).to(device)

    # Inference
    with torch.inference_mode():
        # Get logits, squeeze to remove batch dimension, convert to float
        score = model(pixel_values).logits.squeeze().float().cpu().item()

    return score

def atomic_copy(src, dst_folder, new_filename):
    """
    Safely copies file to temp name then renames.
    Prevents corrupted files if script is interrupted.
    """
    temp_name = f".tmp_{new_filename}"
    temp_path = os.path.join(dst_folder, temp_name)
    final_path = os.path.join(dst_folder, new_filename)

    try:
        shutil.copy2(src, temp_path)  # Preserves metadata
        os.rename(temp_path, final_path)
    except Exception as e:
        print(f"Failed to copy {src}: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def main():
    # 1. Setup
    if not os.path.exists(INPUT_FOLDER):
        print(f"Input folder '{INPUT_FOLDER}' not found.")
        return

    device, dtype = setup_device()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 2. Load Model
    model, preprocessor = load_model(device, dtype)

    # 3. Find Images
    print("Scanning for images...")
    image_paths = []
    for root, _, files in os.walk(INPUT_FOLDER):
        for file in files:
            if os.path.splitext(file)[1].lower() in EXTENSIONS:
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images. Scoring...")

    # 4. Score Loop
    scored_images = []
    for img_path in tqdm(image_paths, unit="img"):
        score = get_image_score(img_path, model, preprocessor, device, dtype)
        scored_images.append((score, img_path))

    # 5. Sort and Top-K
    # Sort by score descending (High to Low)
    scored_images.sort(key=lambda x: x[0], reverse=True)
    
    # Slice the top K
    best_images = scored_images[:TOP_K]
    
    print(f"\nTop {len(best_images)} selected. Range: {best_images[-1][0]:.2f} - {best_images[0][0]:.2f}")
    print(f"Copying to {OUTPUT_FOLDER}...")

    # 6. Copy
    for score, src_path in tqdm(best_images, unit="copy"):
        original_name = os.path.basename(src_path)
        # Format: 9.55_filename.png
        new_name = f"{score:.2f}_{original_name}"
        
        atomic_copy(src_path, OUTPUT_FOLDER, new_name)

    print("\nDone!")

if __name__ == "__main__":
    main()