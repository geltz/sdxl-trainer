import os
import shutil
from pathlib import Path

# Configuration
source_dir = "."  # Current directory
target_dir = "nonmatching"
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# Create target directory if needed
os.makedirs(target_dir, exist_ok=True)

# Get all txt files
txt_files = Path(source_dir).glob("*.txt")

for txt_file in txt_files:
    stem = txt_file.stem
    
    # Check if matching image exists
    has_match = any(Path(source_dir, f"{stem}{ext}").exists() 
                    for ext in image_extensions)
    
    if not has_match:
        shutil.move(str(txt_file), target_dir)
        print(f"Moved: {txt_file.name}")