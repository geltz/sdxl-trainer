import shutil
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

SOURCE_FOLDER = "."
OUTPUT_FOLDER = "topk_color"
DOWNSCALE_SIZE = 96


def score_image_pure_pillow(im: Image.Image) -> float:
    """Score based on saturation and color variance"""
    im = im.convert("RGB")
    im = im.resize((DOWNSCALE_SIZE, DOWNSCALE_SIZE))
    pixels = list(im.getdata())
    
    if not pixels:
        return -1e9

    total_sat = 0.0
    total_var = 0.0
    
    for r, g, b in pixels:
        mx = max(r, g, b)
        mn = min(r, g, b)
        total_sat += (mx - mn)
        
        gray = (r + g + b) / 3.0
        total_var += abs(r - gray) + abs(g - gray) + abs(b - gray)

    n = len(pixels)
    avg_sat = total_sat / n / 255.0
    avg_var = total_var / n / 255.0
    
    return 100 * avg_sat + 30 * avg_var


def score_image_numpy(im: Image.Image) -> float:
    """Score based on colorfulness, saturation, minus edginess"""
    im = im.convert("RGB")
    im = im.resize((DOWNSCALE_SIZE, DOWNSCALE_SIZE))
    arr = np.asarray(im).astype("float32")

    if arr.size == 0:
        return -1e9

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Saturation
    mx = np.max(arr, axis=2)
    mn = np.min(arr, axis=2)
    sat = (mx - mn) / 255.0
    avg_sat = float(np.mean(sat))

    # Colorfulness (Hasler metric)
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg = float(np.std(rg))
    std_yb = float(np.std(yb))
    mean_rg = float(np.mean(rg))
    mean_yb = float(np.mean(yb))
    colorfulness = (std_rg**2 + std_yb**2) ** 0.5 + 0.3 * ((mean_rg**2 + mean_yb**2) ** 0.5)

    # Edginess (penalize screenshots/text)
    dx = np.abs(np.diff(arr, axis=1)).mean()
    dy = np.abs(np.diff(arr, axis=0)).mean()
    edgeiness = (dx + dy) / 255.0

    return 1.0 * colorfulness + 80.0 * avg_sat - 25.0 * edgeiness


def image_score(path: Path) -> float:
    """Calculate score for an image"""
    if not PIL_AVAILABLE:
        return 0.0
    
    try:
        with Image.open(path) as im:
            if NP_AVAILABLE:
                return score_image_numpy(im)
            else:
                return score_image_pure_pillow(im)
    except Exception:
        return -1e9


def main():
    src = Path(SOURCE_FOLDER)
    out = src / OUTPUT_FOLDER
    out.mkdir(exist_ok=True)

    # Find all images
    image_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    images = [p for p in src.iterdir() if p.suffix.lower() in image_exts]

    if not images:
        print("No images found.")
        return

    print(f"Found {len(images)} images.")
    
    # Prompt for TOP_K
    while True:
        try:
            top_k = int(input(f"How many top images to select? (1-{len(images)}): "))
            if 1 <= top_k <= len(images):
                break
            print(f"Please enter a number between 1 and {len(images)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            return

    print("Scoring...")

    # Score all images
    scored = [(image_score(img), img) for img in images]
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [img for _, img in scored[:top_k]]

    # Copy selected images and their captions
    print(f"Copying {len(selected)} best images to {out}...")
    
    for img in selected:
        shutil.copy2(img, out / img.name)
        txt = img.with_suffix(".txt")
        if txt.exists():
            shutil.copy2(txt, out / txt.name)

    print("Done.")


if __name__ == "__main__":
    main()
