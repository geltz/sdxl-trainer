import os
import shutil
from pathlib import Path

# Image extensions to look for
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# Get all image filenames (without extension)
image_files = set()
for f in os.listdir('.'):
    path = Path(f)
    if path.suffix.lower() in IMAGE_EXTS:
        image_files.add(path.stem)

# Create output directory
os.makedirs('nonmatching_embed', exist_ok=True)

# Check embeddings cache
cache_dir = '.precomputed_embeddings_cache'
if os.path.exists(cache_dir):
    for f in os.listdir(cache_dir):
        if f.endswith('.pt'):
            # Get filename without .pt extension
            embed_name = Path(f).stem
            
            # If no matching image, move it
            if embed_name not in image_files:
                src = os.path.join(cache_dir, f)
                dst = os.path.join('nonmatching_embed', f)
                shutil.move(src, dst)
                print(f"Moved: {f}")