import shutil
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False

SOURCE_FOLDER = "."
OUTPUT_FOLDER = "topk_color"
DOWNSCALE_SIZE = 128


def score_image_torch(im: Image.Image) -> float:
    """Enhanced scoring for anime images"""
    im = im.convert("RGB")
    orig_w, orig_h = im.size
    im = im.resize((DOWNSCALE_SIZE, DOWNSCALE_SIZE))
    arr = torch.tensor(np.asarray(im), dtype=torch.float32, device=DEVICE)

    if arr.numel() == 0:
        return -1e9

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    # Saturation (higher = more vibrant)
    mx = torch.max(arr, dim=2).values
    mn = torch.min(arr, dim=2).values
    sat = (mx - mn) / 255.0
    avg_sat = float(sat.mean())
    sat_std = float(sat.std())

    # Colorfulness
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg = float(rg.std())
    std_yb = float(yb.std())
    mean_rg = float(rg.mean())
    mean_yb = float(yb.mean())
    colorfulness = (std_rg**2 + std_yb**2) ** 0.5 + 0.3 * ((mean_rg**2 + mean_yb**2) ** 0.5)

    # Color diversity (unique color bins)
    quant = (arr / 32).long()
    unique_colors = len(torch.unique(quant[:, :, 0] * 64 + quant[:, :, 1] * 8 + quant[:, :, 2]))
    color_diversity = unique_colors / (DOWNSCALE_SIZE * DOWNSCALE_SIZE)

    # Brightness balance (penalize too dark or too bright)
    brightness = arr.mean() / 255.0
    brightness_penalty = abs(brightness - 0.55) * 50

    # Contrast
    contrast = float((mx - mn).mean()) / 255.0

    # Edge detection (moderate edges = good detail)
    dx = torch.abs(torch.diff(arr, dim=1)).mean()
    dy = torch.abs(torch.diff(arr, dim=0)).mean()
    edginess = float((dx + dy) / 255.0)
    
    # Penalize extreme blur or noise
    edge_penalty = 0
    if edginess < 0.05:  # too blurry
        edge_penalty = (0.05 - edginess) * 200
    elif edginess > 0.4:  # too noisy
        edge_penalty = (edginess - 0.4) * 150

    # Resolution bonus (prefer higher res originals)
    res_score = min((orig_w * orig_h) / (1024 * 1024), 2.0) * 5

    # White/black dominance penalty
    white_ratio = float((arr > 240).all(dim=2).float().mean())
    black_ratio = float((arr < 15).all(dim=2).float().mean())
    extreme_penalty = (white_ratio + black_ratio) * 80

    score = (
        2.0 * colorfulness +
        120.0 * avg_sat +
        30.0 * sat_std +
        40.0 * color_diversity +
        50.0 * contrast +
        res_score -
        brightness_penalty -
        edge_penalty -
        extreme_penalty
    )

    return score


def score_image_numpy(im: Image.Image) -> float:
    """Enhanced scoring with numpy"""
    im = im.convert("RGB")
    orig_w, orig_h = im.size
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
    sat_std = float(np.std(sat))

    # Colorfulness
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg = float(np.std(rg))
    std_yb = float(np.std(yb))
    mean_rg = float(np.mean(rg))
    mean_yb = float(np.mean(yb))
    colorfulness = (std_rg**2 + std_yb**2) ** 0.5 + 0.3 * ((mean_rg**2 + mean_yb**2) ** 0.5)

    # Color diversity
    quant = (arr / 32).astype(int)
    unique_colors = len(np.unique(quant[:, :, 0] * 64 + quant[:, :, 1] * 8 + quant[:, :, 2]))
    color_diversity = unique_colors / (DOWNSCALE_SIZE * DOWNSCALE_SIZE)

    # Brightness
    brightness = arr.mean() / 255.0
    brightness_penalty = abs(brightness - 0.55) * 50

    # Contrast
    contrast = float((mx - mn).mean()) / 255.0

    # Edges
    dx = np.abs(np.diff(arr, axis=1)).mean()
    dy = np.abs(np.diff(arr, axis=0)).mean()
    edginess = (dx + dy) / 255.0
    
    edge_penalty = 0
    if edginess < 0.05:
        edge_penalty = (0.05 - edginess) * 200
    elif edginess > 0.4:
        edge_penalty = (edginess - 0.4) * 150

    # Resolution
    res_score = min((orig_w * orig_h) / (1024 * 1024), 2.0) * 5

    # Extremes
    white_ratio = float(np.mean(np.all(arr > 240, axis=2)))
    black_ratio = float(np.mean(np.all(arr < 15, axis=2)))
    extreme_penalty = (white_ratio + black_ratio) * 80

    score = (
        2.0 * colorfulness +
        120.0 * avg_sat +
        30.0 * sat_std +
        40.0 * color_diversity +
        50.0 * contrast +
        res_score -
        brightness_penalty -
        edge_penalty -
        extreme_penalty
    )

    return score


def score_image_pure_pillow(im: Image.Image) -> float:
    """Fallback scoring with Pillow only"""
    im = im.convert("RGB")
    orig_w, orig_h = im.size
    im = im.resize((DOWNSCALE_SIZE, DOWNSCALE_SIZE))
    pixels = list(im.getdata())
    
    if not pixels:
        return -1e9

    total_sat = 0.0
    total_var = 0.0
    brightness_sum = 0.0
    
    for r, g, b in pixels:
        mx = max(r, g, b)
        mn = min(r, g, b)
        total_sat += (mx - mn)
        
        gray = (r + g + b) / 3.0
        brightness_sum += gray
        total_var += abs(r - gray) + abs(g - gray) + abs(b - gray)

    n = len(pixels)
    avg_sat = total_sat / n / 255.0
    avg_var = total_var / n / 255.0
    brightness = brightness_sum / n / 255.0
    brightness_penalty = abs(brightness - 0.55) * 50
    res_score = min((orig_w * orig_h) / (1024 * 1024), 2.0) * 5
    
    return 120 * avg_sat + 40 * avg_var + res_score - brightness_penalty


def image_score(path: Path) -> float:
    """Calculate score for an image"""
    if not PIL_AVAILABLE:
        return 0.0
    
    try:
        with Image.open(path) as im:
            if TORCH_AVAILABLE:
                return score_image_torch(im)
            elif NP_AVAILABLE:
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
    if TORCH_AVAILABLE:
        print(f"Using PyTorch on {DEVICE}")
    elif NP_AVAILABLE:
        print("Using NumPy")
    else:
        print("Using pure Pillow")
    
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

    # Score all images with progress
    scored = []
    for i, img in enumerate(images, 1):
        if i % 1000 == 0:
            print(f"  {i}/{len(images)}...")
        scored.append((image_score(img), img))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [img for _, img in scored[:top_k]]

    # Copy selected images and captions
    print(f"Copying {len(selected)} best images to {out}...")
    
    for img in selected:
        shutil.copy2(img, out / img.name)
        txt = img.with_suffix(".txt")
        if txt.exists():
            shutil.copy2(txt, out / txt.name)

    print("Done.")


if __name__ == "__main__":
    main()