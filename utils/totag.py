import os
import shutil
from pathlib import Path

def move_untagged_images(source_dir='.', target_dir='totag'):
    """Move images without matching .txt files to totag folder."""
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    target_path.mkdir(exist_ok=True)
    
    moved_count = 0
    
    for file in source_path.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            txt_file = file.with_suffix('.txt')
            
            # Move if no matching .txt exists
            if not txt_file.exists():
                dest = target_path / file.name
                shutil.move(str(file), str(dest))
                moved_count += 1
                print(f"Moved: {file.name}")
    
    print(f"\nTotal moved: {moved_count}")

if __name__ == '__main__':
    move_untagged_images()