import os
import json
import sys
import random
import threading
import webbrowser
import torch
from PIL import Image, ImageFile
from flask import Flask, render_template_string, send_from_directory, jsonify

# ================= CONFIGURATION =================
INPUT_FOLDER = r"."  # Folder containing your images
CACHE_FILE = "image_scores_cache.json"
PORT = 5000
# =================================================

app = Flask(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------------------------------
# 1. MODEL & SCORING LOGIC
# -------------------------------------------------

def get_siglip_model():
    try:
        from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
    except ImportError:
        print("Please run: pip install aesthetic-predictor-v2-5")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Auto-detect precision
    dtype = torch.float32
    if device == "cuda":
        try:
            torch.zeros(1).to(device, dtype=torch.bfloat16)
            dtype = torch.bfloat16
            print(">> Using CUDA bfloat16")
        except TypeError:
            dtype = torch.float16
            print(">> Using CUDA float16")
    
    print(">> Loading Model...")
    model, preprocessor = convert_v2_5_from_siglip(low_cpu_mem_usage=True, trust_remote_code=True)
    model = model.to(dtype).to(device)
    model.eval()
    return model, preprocessor, device, dtype

def scan_and_score(folder):
    """
    Scans folder, loads cache, scores new images, saves cache.
    Returns a list of dicts: [{'name': 'img.png', 'score': 9.2}, ...]
    """
    # 1. Load Cache
    scores = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                scores = json.load(f)
            print(f">> Loaded {len(scores)} cached scores.")
        except:
            print(">> Cache corrupted, starting fresh.")

    # 2. Find Images
    valid_ext = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in valid_ext]
    
    # Filter images that need scoring
    to_score = [f for f in image_files if f not in scores]

    if to_score:
        print(f">> Scoring {len(to_score)} new images...")
        model, preprocessor, device, dtype = get_siglip_model()
        
        count = 0
        total = len(to_score)
        
        for filename in to_score:
            path = os.path.join(folder, filename)
            try:
                img = Image.open(path).convert("RGB")
                inputs = preprocessor(images=img, return_tensors="pt")
                pixel_values = inputs.pixel_values.to(dtype).to(device)
                
                with torch.inference_mode():
                    score = model(pixel_values).logits.squeeze().float().cpu().item()
                
                scores[filename] = round(score, 4)
            except Exception as e:
                print(f"Error {filename}: {e}")
                scores[filename] = 0.0
            
            count += 1
            if count % 10 == 0:
                print(f"   Progress: {count}/{total}")

        # Save Cache
        with open(CACHE_FILE, 'w') as f:
            json.dump(scores, f, indent=2)
    
    # Return list sorted by score (Highest first)
    # Also filter out files that might have been deleted from disk
    final_list = []
    for f in image_files:
        if f in scores:
            final_list.append({'name': f, 'score': scores[f]})
    
    final_list.sort(key=lambda x: x['score'], reverse=True)
    return final_list

# -------------------------------------------------
# 2. WEB SERVER ENDPOINTS
# -------------------------------------------------

# HTML TEMPLATE (Embedded for single-file convenience)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Aesthetic Visualizer</title>
    <style>
        body { 
            margin: 0; 
            background: #111; 
            color: #eee; 
            font-family: sans-serif; 
            overflow-x: hidden; /* We handle scroll inside container */
        }
        
        #header {
            position: fixed; top: 0; left: 0; right: 0; height: 60px;
            background: rgba(0,0,0,0.8); z-index: 1000;
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 20px; backdrop-filter: blur(5px);
            border-bottom: 1px solid #333;
        }

        #timeline-container {
            position: relative;
            width: 100%;
            height: 100vh;
            overflow-x: auto;
            overflow-y: hidden;
            perspective: 1000px;
        }

        /* The long strip representing 0 to 10 scores */
        #score-track {
            position: relative;
            height: 100%;
            /* Width will be set dynamically based on density */
            min-width: 120vw; 
            background: linear-gradient(to right, #220000 0%, #111 50%, #002200 100%);
        }

        .img-node {
            position: absolute;
            top: 50%;
            transform: translate(-50%, -50%);
            width: 60px; 
            height: auto;
            border-radius: 4px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.5);
            cursor: pointer;
            transition: z-index 0.1s;
            will-change: transform; /* Performance optimization */
            object-fit: cover;
        }
        
        .score-marker {
            position: absolute; bottom: 20px; 
            font-size: 100px; font-weight: bold; color: rgba(255,255,255,0.05);
            transform: translateX(-50%);
            pointer-events: none;
        }

        /* Info Box */
        #info-box {
            position: fixed; bottom: 20px; left: 20px;
            background: rgba(0,0,0,0.8); padding: 10px;
            border-radius: 5px; z-index: 1000;
            pointer-events: none; opacity: 0; transition: opacity 0.2s;
        }
    </style>
</head>
<body>

<div id="header">
    <h2>Aesthetic Visualizer (SigLIP)</h2>
    <div id="stats">Loading...</div>
</div>

<div id="timeline-container">
    <div id="score-track">
        <!-- Images injected here -->
    </div>
</div>

<div id="info-box">
    <div id="info-name"></div>
    <div id="info-score" style="color:#0f0; font-weight:bold"></div>
</div>

<script>
    const track = document.getElementById('score-track');
    const container = document.getElementById('timeline-container');
    const infoBox = document.getElementById('info-box');
    const infoName = document.getElementById('info-name');
    const infoScore = document.getElementById('info-score');
    
    let images = [];
    
    // CONFIG FOR ANIMATION
    const BASE_SIZE = 80;       // Normal width
    const MAX_SCALE = 4.5;      // How much bigger on hover (4.5x)
    const EFFECT_RADIUS = 250;  // Mouse distance to trigger effect

    async function loadData() {
        const response = await fetch('/api/data');
        const data = await response.json();
        document.getElementById('stats').innerText = data.length + " Images Scored";
        render(data);
    }

    function render(data) {
        // 1. Determine track width based on density to prevent overcrowding
        // We map score 0-10 to 0-100% of width
        // But if we have many images, we want the track longer.
        const trackWidth = Math.max(window.innerWidth, data.length * 20); 
        track.style.width = trackWidth + "px";

        // 2. Add Background Score Markers (0, 1, ... 10)
        for(let i=0; i<=10; i++) {
            let m = document.createElement('div');
            m.className = 'score-marker';
            m.innerText = i;
            m.style.left = (i/10 * 100) + "%";
            track.appendChild(m);
        }

        // 3. Create Image Nodes
        data.forEach(item => {
            const img = document.createElement('img');
            img.src = "/images/" + encodeURIComponent(item.name);
            img.className = 'img-node';
            img.dataset.score = item.score;
            img.dataset.name = item.name;

            // X Position based on score (0 to 10)
            // We map 0-10 to 5%-95% of the screen to add padding
            const pct = (item.score / 10) * 90 + 5; 
            img.style.left = pct + "%";

            // Y Position: Random Scatter to create a "Cloud" or "Galaxy" effect
            // Centered vertically with +/- spread
            const scatter = (Math.random() - 0.5) * 70; // 70% height spread
            img.style.top = (50 + scatter) + "%";
            
            // Random slight z-rotation for organic feel
            img.style.transform = `translate(-50%, -50%) rotate(${(Math.random()-0.5)*15}deg)`;

            track.appendChild(img);
            images.push({
                el: img,
                baseX: pct, // We'll need to recalculate pixel pos on resize, but simple is fine for now
                baseY: 50 + scatter,
                currentScale: 1
            });
            
            // Hover info
            img.addEventListener('mouseenter', () => {
                infoBox.style.opacity = 1;
                infoName.innerText = item.name;
                infoScore.innerText = "Score: " + item.score;
                img.style.zIndex = 999; // Bring to very front
            });
            img.addEventListener('mouseleave', () => {
                infoBox.style.opacity = 0;
                img.style.zIndex = ""; // Reset
            });
        });
    }

    // ----------------------------------------------------
    // THE PHYSICS LOOP (The "Mac Dock" Effect)
    // ----------------------------------------------------
    let mouseX = -9999;
    let mouseY = -9999;

    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });
    
    // Also handle scrolling (mouse position is relative to viewport)
    container.addEventListener('scroll', () => {
        // Trigger update
    });

    function animate() {
        requestAnimationFrame(animate);

        // Optimize: get bounding rects is expensive, 
        // but we need screen coordinates vs mouse coordinates.
        
        images.forEach(obj => {
            const rect = obj.el.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;

            // Distance from mouse to image center
            const dist = Math.sqrt(Math.pow(mouseX - centerX, 2) + Math.pow(mouseY - centerY, 2));

            let targetScale = 1;
            
            if (dist < EFFECT_RADIUS) {
                // Calculate Scale: Linear or Cosine interpolation
                // 1 at 0 distance, 0 at EFFECT_RADIUS
                const amount = 1 - (dist / EFFECT_RADIUS);
                
                // Easing creates smooth curve
                const eased = Math.pow(amount, 2); 
                targetScale = 1 + (MAX_SCALE - 1) * eased;
                
                // Increase z-index if growing
                obj.el.style.zIndex = Math.floor(targetScale * 10);
            } else {
                obj.el.style.zIndex = 1;
            }

            // Apply transform
            // We keep the rotation we set initially? No, let's straighten it out as it grows for better viewing
            if (targetScale > 1.1) {
                obj.el.style.transform = `translate(-50%, -50%) scale(${targetScale})`;
                obj.el.style.boxShadow = `0 10px 30px rgba(0,0,0,0.8)`;
            } else {
                // Revert to chaos state (optional, or just keep flat)
                 obj.el.style.transform = `translate(-50%, -50%) scale(1)`;
                 obj.el.style.boxShadow = `0 4px 10px rgba(0,0,0,0.5)`;
            }
        });
    }

    loadData();
    animate();

</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def data():
    # Check/Generate scores
    scored_data = scan_and_score(INPUT_FOLDER)
    return jsonify(scored_data)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(INPUT_FOLDER, filename)

def open_browser():
    webbrowser.open(f'http://127.0.0.1:{PORT}')

if __name__ == "__main__":
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Folder '{INPUT_FOLDER}' does not exist.")
        print("Please edit the script to point to your image folder.")
    else:
        print(f"Starting Visualizer for: {INPUT_FOLDER}")
        # Open browser after slight delay to ensure server is up
        threading.Timer(1.5, open_browser).start()
        app.run(port=PORT, debug=False)