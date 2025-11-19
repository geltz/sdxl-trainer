import os
import re
import gc
import json
import math
import random
import inspect
import warnings
from pathlib import Path
from collections import defaultdict, deque

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image, TiffImagePlugin, ImageFile
from tqdm.auto import tqdm
from torchvision import transforms
from safetensors.torch import save_file, load_file
from diffusers import (
    StableDiffusionXLPipeline,
    DDPMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers.optimization import Adafactor

import config as default_config
from optimizer.raven import RavenAdamW

import multiprocessing
import argparse
import numpy as np

import config as default_config

from optimizer.lora import (
    inject_lora_peft,
    extract_lora_state_dict_comfy_peft,
)

from optimizer.locon import inject_locon_peft, extract_locon_state_dict_comfy_peft

from safetensors.torch import save_file as save_safetensors

class CachingDataset(Dataset):
    def __init__(self, file_paths, config):
        self.file_paths = file_paths
        self.config = config
        # Re-instantiate calculator here to ensure it's safe for workers
        self.calc = ResolutionCalculator(
            config.TARGET_PIXEL_AREA, 
            64, 
            config.SHOULD_UPSCALE, 
            getattr(config, "MAX_AREA_TOLERANCE", 1.1)
        )
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                w, h = im.size
                
            tw, th = self.calc.calculate_resolution(w, h)
            im = resize_to_fit(im, tw, th)
            img_tensor = self.transform(im)
            
            cp = path.with_suffix(".txt")
            if cp.exists():
                caption = cp.read_text(encoding="utf-8").strip() or path.stem.replace("_", " ")
            else:
                caption = path.stem.replace("_", " ")
            
            # Tag dropout logic
            whitelist_str = getattr(self.config, "TAG_DROPOUT_WHITELIST", "")
            whitelist = [tag.strip() for tag in whitelist_str.split(',') if tag.strip()]
            dropout_rate = getattr(self.config, "TAG_DROPOUT_RATE", 0.0)
            caption = apply_tag_dropout(caption, dropout_rate, whitelist, log_samples=False)

            return {
                "img_tensor": img_tensor,
                "caption": caption,
                "original_size": (w, h),
                "target_size": (tw, th),
                "path": str(path),
                "stem": path.stem
            }
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

def caching_collate(batch):
    return [b for b in batch if b is not None]


def convert_lora_pt_to_safetensors(pt_path):
    """Convert LoRA .pt to .safetensors immediately after saving."""
    try:
        pt_path = Path(pt_path)
        output_path = pt_path.with_suffix('.safetensors')
        
        # Load and convert
        lora_state = torch.load(pt_path, map_location="cpu")
        save_safetensors(lora_state, output_path)
        
        # Report sizes
        pt_size = pt_path.stat().st_size / (1024 * 1024)
        st_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  Converted to safetensors: {output_path.name} ({st_size:.2f} MB)")
        
        return output_path
    except Exception as e:
        print(f"  Warning: Could not convert to safetensors: {e}")
        return None

# ----- model path helpers (must be defined before main) -----
def normalize_model_path(pathlike):
    """Return an existing absolute str path to the checkpoint."""
    from pathlib import Path

    p = pathlike if isinstance(pathlike, Path) else Path(str(pathlike))

    # if a directory was given, assume model.safetensors inside it
    if p.is_dir():
        candidate = p / "model.safetensors"
        if candidate.exists():
            p = candidate

    if not p.exists():
        # try relative to this file
        here = Path(__file__).resolve().parent
        candidate = (here / p).resolve()
        if candidate.exists():
            p = candidate
        else:
            raise FileNotFoundError(f"Checkpoint file not found after resolving: {p}")

    return str(p.resolve())

def apply_tag_dropout(caption: str, dropout_rate: float, whitelist: list, log_samples: bool = True) -> str:
    if dropout_rate <= 0 or not caption:
        return caption
    
    tags = [t.strip() for t in caption.split(',')]
    if len(tags) <= 1:
        return caption
    
    def matches_whitelist(tag: str, whitelist: list) -> bool:
        tag_lower = tag.lower()
        for pattern in whitelist:
            pattern_lower = pattern.lower()
            if '*' in pattern_lower:
                regex_pattern = '^' + pattern_lower.replace('*', '.*') + '$'
                if re.match(regex_pattern, tag_lower):
                    return True
            elif tag_lower == pattern_lower:
                return True
        return False
    
    kept_tags = []
    dropped = []
    
    for tag in tags:
        # If whitelisted, always keep
        if matches_whitelist(tag, whitelist):
            kept_tags.append(tag)
        # Otherwise, apply dropout randomly
        elif random.random() > dropout_rate:
            kept_tags.append(tag)
        else:
            dropped.append(tag)
    
    # Ensure at least one tag remains
    if not kept_tags:
        kept_tags = [random.choice(tags)]
    
    result = ', '.join(kept_tags)
    
    if log_samples and hasattr(apply_tag_dropout, '_sample_count'):
        apply_tag_dropout._sample_count += 1
        if apply_tag_dropout._sample_count <= 5:
            print(f"\n[Tag Dropout Sample #{apply_tag_dropout._sample_count}]")
            print(f"  Original: {caption}")
            print(f"  Dropped: {dropped if dropped else 'none'}")
            print(f"  Result: {result}")
    
    return result

# Initialize counter
apply_tag_dropout._sample_count = 0

def get_training_model_path(config):
    # config is already JSON-merged by TrainingConfig.__init__()
    if getattr(config, "RESUME_TRAINING", False) and getattr(config, "RESUME_MODEL_PATH", ""):
        raw = config.RESUME_MODEL_PATH
    elif getattr(config, "MODEL_PATH", ""):
        raw = config.MODEL_PATH
    else:
        raw = getattr(config, "SINGLE_FILE_CHECKPOINT_PATH", "./model.safetensors")
    return normalize_model_path(raw)

def apply_reflection_padding_to_unet(unet, enabled: bool = False):
    if not enabled:
        return

    import torch.nn as nn

    # Build a flat dict so we can look up parents by name once
    module_dict = dict(unet.named_modules())

    to_modify = []

    # First: collect all Conv2d layers that actually use padding
    for name, module in module_dict.items():
        if not isinstance(module, nn.Conv2d):
            continue

        # Conv2d.padding can be int or (h, w)
        if isinstance(module.padding, tuple):
            pad_h, pad_w = module.padding
        else:
            pad_h = pad_w = int(module.padding)

        if pad_h == 0 and pad_w == 0:
            continue  # nothing to do

        to_modify.append((name, module, pad_h, pad_w))

    # Second: wrap them with ReflectionPad2d
    for name, conv, pad_h, pad_w in to_modify:
        # Remove internal conv padding – we'll do it explicitly via ReflectionPad2d
        conv.padding = (0, 0)

        # Find parent module and attribute name
        parent_name, _, child_name = name.rpartition(".")
        parent = unet if parent_name == "" else module_dict[parent_name]

        # ReflectionPad2d expects (left, right, top, bottom)
        reflect_padding = (pad_w, pad_w, pad_h, pad_h)

        wrapped = nn.Sequential(
            nn.ReflectionPad2d(reflect_padding),
            conv,
        )

        setattr(parent, child_name, wrapped)

    if to_modify:
        print(f"INFO: Applied reflection padding to {len(to_modify)} Conv2d layers")

# ==========================
# Global settings
# ==========================
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=TiffImagePlugin.__name__,
    message="Corrupt EXIF data",
)
Image.MAX_IMAGE_PIXELS = 190_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==========================
# Utility functions
# ==========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"INFO: Set random seed to {seed}")


def filter_scheduler_config(cfg: dict, scheduler_class):
    sig = inspect.signature(scheduler_class.__init__).parameters
    return {k: v for k, v in cfg.items() if k in sig}


def rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    device, dtype = betas.device, betas.dtype
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_cumprod)

    a0 = alphas_bar_sqrt[0].clone()
    aT = alphas_bar_sqrt[-1].clone()

    alphas_bar_sqrt = alphas_bar_sqrt - aT
    alphas_bar_sqrt = alphas_bar_sqrt * (a0 / (a0 - aT))

    alphas_bar = alphas_bar_sqrt ** 2
    alphas = alphas_bar / F.pad(alphas_bar[:-1], (1, 0), value=1.0)
    return (1.0 - alphas).to(device=device, dtype=dtype)


# ==========================
# Config wrapper
# ==========================

class TrainingConfig:
    def __init__(self):
        # 1) load defaults from config.py
        for k, v in default_config.__dict__.items():
            if not k.startswith("__"):
                setattr(self, k, v)

        # 2) load external JSON if present
        self._load_user_json()

        # 3) normalize types against config.py
        self._fix_types()

        self.compute_dtype = (
            torch.bfloat16 if self.MIXED_PRECISION == "bfloat16" else torch.float16
        )
        
        # 4) Set VAE shift factor (for Flux/RF models)
        if not hasattr(self, 'VAE_SHIFT_FACTOR'):
            self.VAE_SHIFT_FACTOR = 0.0
        
        # 5) SDXL VAE scaling
        if not hasattr(self, 'VAE_SCALING_FACTOR'):
            self.VAE_SCALING_FACTOR = 0.13025

    def _load_user_json(self):
        import argparse
        from pathlib import Path

        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--config", type=str, default=None)
        args, _ = parser.parse_known_args()

        cfg_path = None

        if args.config:
            cfg_path = Path(args.config)
        else:
            here = Path(__file__).resolve().parent          # D:\finetune\trainer
            parent_dir = here.parent                        # D:\finetune
            cwd = Path.cwd()                                # probably D:\finetune\trainer

            search_roots = [here, parent_dir, cwd]

            for root in search_roots:
                for name in ("koeri.json", "config.json", "trainer.json"):
                    cand = root / name
                    if cand.exists():
                        cfg_path = cand
                        print(f"INFO: Auto-detected external JSON config: {cand}")
                        break
                if cfg_path:
                    break

        if not cfg_path:
            print("INFO: No external JSON config. Using config.py only.")
            return
        if not cfg_path.exists():
            print(f"WARNING: Config {cfg_path} not found. Skipping.")
            return

        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        for k, v in data.items():
            setattr(self, k, v)

        # allow JSONs that only set MODEL_PATH
        if getattr(self, "MODEL_PATH", None) and not getattr(self, "SINGLE_FILE_CHECKPOINT_PATH", None):
            self.SINGLE_FILE_CHECKPOINT_PATH = self.MODEL_PATH

        print(f"INFO: Loaded overrides from {cfg_path}")

    def _fix_types(self):
        print("INFO: Normalizing config types...")
        base = default_config  # use the module we imported at the top
        for k, v in list(self.__dict__.items()):
            # special case: string -> list
            if k == "UNET_EXCLUDE_TARGETS" and isinstance(v, str):
                setattr(self, k, [s.strip() for s in v.split(",") if s.strip()])
                continue

            dv = getattr(base, k, None)
            if dv is None:
                continue
            if isinstance(v, type(dv)):
                continue
            if isinstance(dv, bool) and isinstance(v, str):
                setattr(self, k, v.lower() in ("1", "true", "yes", "y"))
                continue
            try:
                if isinstance(dv, int):
                    setattr(self, k, int(float(v)))
                else:
                    setattr(self, k, type(dv)(v))
            except Exception:
                print(f"WARNING: Could not coerce {k}={v!r} to {type(dv)}; keeping default.")
                setattr(self, k, dv)
        print("INFO: Config is ready.")

# ==========================
# Dataset + caching helpers
# ==========================
class ResolutionCalculator:
    def __init__(self, target_area, stride=64, should_upscale=False, max_area_tolerance=1.1):
        self.target_area = target_area
        self.stride = stride
        self.should_upscale = should_upscale
        self.max_area = target_area * max_area_tolerance

    def calculate_resolution(self, width, height):
        aspect = width / height
        if not self.should_upscale:
            h = int(math.sqrt(self.target_area / aspect) // self.stride) * self.stride
            w = int(h * aspect // self.stride) * self.stride
            return max(w, self.stride), max(h, self.stride)
        # upscale branch (kept from original)
        current_area = width * height
        scale = math.sqrt(self.target_area / current_area)
        w = int((width * scale) // self.stride) * self.stride
        h = int((height * scale) // self.stride) * self.stride
        return max(w, self.stride), max(h, self.stride)


def resize_to_fit(image, target_w, target_h):
    w, h = image.size
    if w / target_w < h / target_h:
        w_new, h_new = target_w, int(h * target_w / w)
    else:
        w_new, h_new = int(w * target_h / h), target_h
    return image.resize((w_new, h_new), Image.Resampling.LANCZOS).crop(
        (
            (w_new - target_w) // 2,
            (h_new - target_h) // 2,
            (w_new + target_w) // 2,
            (h_new + target_h) // 2,
        )
    )


def check_if_caching_needed(config: TrainingConfig):
    needs = False
    for ds in config.INSTANCE_DATASETS:
        root = Path(ds["path"])
        if not root.exists():
            print(f"WARNING: Dataset path does not exist: {root}")
            needs = True
            continue
        imgs = [p for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"] for p in root.rglob(f"*{ext}")]
        if not imgs:
            print(f"WARNING: No images found in: {root}")
            needs = True
            continue
        cache_dir = root / ".precomputed_embeddings_cache"
        if not cache_dir.exists():
            needs = True
            continue
        cached = {p.stem for p in cache_dir.glob("*.pt")}
        if any(p.stem not in cached for p in imgs):
            needs = True
    return needs


def load_vae_only(config: TrainingConfig, device):
    vae_path = getattr(config, "VAE_PATH", None)
    if not vae_path or not Path(vae_path).exists():
        return None
    print(f"INFO: Loading dedicated VAE from {vae_path}")
    vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float32).to(device)
    # vae.enable_tiling() # no tiling
    return vae


def compute_chunked_text_embeddings(captions, t1, t2, te1, te2, device):
    prompt_embeds, pooled_embeds = [], []
    for cap in captions:
        with torch.no_grad():
            i1 = t1(cap, padding="max_length", max_length=t1.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
            i2 = t2(cap, padding="max_length", max_length=t2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)
            e1 = te1(i1, output_hidden_states=True).hidden_states[-2]
            out2 = te2(i2, output_hidden_states=True)
            emb = torch.cat((e1, out2.hidden_states[-2]), dim=-1)
            pooled = out2[0]
        prompt_embeds.append(emb)
        pooled_embeds.append(pooled)
    return torch.cat(prompt_embeds), torch.cat(pooled_embeds)

def precompute_and_cache_latents(config: TrainingConfig, t1, t2, te1, te2, vae, device):
    # FIX: Enable tiling. This cuts VRAM usage drastically (required for 12GB cards)
    vae.enable_tiling()
    
    # Ensure models are in eval mode
    vae.eval()
    te1.eval()
    te2.eval()

    for ds in config.INSTANCE_DATASETS:
        root = Path(ds["path"])
        if not root.exists(): continue
        
        img_paths = [p for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"] for p in root.rglob(f"*{ext}")]
        cache_dir = root / ".precomputed_embeddings_cache"
        cache_dir.mkdir(exist_ok=True)
        
        cached = {p.stem for p in cache_dir.glob("*.pt")}
        todo = [p for p in img_paths if p.stem not in cached]
        
        if not todo:
            print(f"INFO: All images in {root} cached.")
            continue
        
        print(f"INFO: Caching {len(todo)} images from {root}")

        # Move Text Encoders to GPU (needed for embedding generation)
        te1.to(device)
        te2.to(device)

        # Use the CachingDataset defined in global scope
        # num_workers=4 allows resizing images on CPU while GPU works
        cache_ds = CachingDataset(todo, config)
        loader = DataLoader(
            cache_ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=4, 
            collate_fn=caching_collate,
            persistent_workers=True
        )

        for batch in tqdm(loader, desc=f"Caching {root}"):
            if not batch: continue
            
            data = batch[0]
            
            img_tensor = data["img_tensor"].unsqueeze(0).to(device, dtype=torch.float32)
            caption = data["caption"]
            
            # 1. Compute Text Embeddings
            embeds, pooled = compute_chunked_text_embeddings([caption], t1, t2, te1, te2, device)
            
            # 2. Compute Latents with Tiling (Prevents OOM)
            with torch.no_grad():
                # diffusers automatically handles slicing when enable_tiling() is on
                latents = vae.encode(img_tensor).latent_dist.mean

                if config.VAE_SHIFT_FACTOR != 0.0:
                    latents = latents - config.VAE_SHIFT_FACTOR

                # Force SDXL scaling
                vae_scale = getattr(vae.config, 'scaling_factor', 0.13025)
                latents = latents * vae_scale
                
            # 3. Save
            torch.save(
                {
                    "original_size": data["original_size"],
                    "target_size": data["target_size"],
                    "embeds": embeds.squeeze(0).cpu(),
                    "pooled": pooled.squeeze(0).cpu(),
                    "latents": latents.squeeze(0).cpu(),
                },
                cache_dir / f"{data['stem']}.pt",
            )

        # Cleanup after this dataset is done
        te1.cpu(); te2.cpu()
        del loader
        del cache_ds
        gc.collect()
        torch.cuda.empty_cache()


class ImageTextLatentDataset(Dataset):
    def __init__(self, config: TrainingConfig):
        self.files = []
        meta = {}
        for ds in config.INSTANCE_DATASETS:
            root = Path(ds["path"])
            cache_dir = root / ".precomputed_embeddings_cache"
            if not cache_dir.exists():
                continue
            flist = list(cache_dir.glob("*.pt"))
            self.files.extend(flist * int(ds.get("repeats", 1)))
            meta_path = root / "metadata.json"
            if meta_path.exists():
                meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
        if not self.files:
            raise ValueError("No cached files found.")
        random.shuffle(self.files)
        self.bucket_keys = [tuple(meta.get(f.stem)) if f.stem in meta else None for f in self.files]
        print(f"Dataset initialized with {len(self.files)} samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        return {
            "latents": data["latents"],
            "embeds": data["embeds"],
            "pooled": data["pooled"],
            "original_sizes": data["original_size"],
            "target_sizes": data["target_size"],
            "latent_path": str(self.files[idx]),
        }


class BucketBatchSampler(Sampler):
    def __init__(self, dataset: ImageTextLatentDataset, batch_size: int, seed: int, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        g = torch.Generator(); g.manual_seed(self.seed)
        idxs = torch.randperm(len(self.indices), generator=g).tolist() if self.shuffle else self.indices
        buckets = defaultdict(list)
        for i in idxs:
            key = self.dataset.bucket_keys[i]
            buckets[key].append(i)
        batches = []
        for key in sorted(buckets.keys(), key=lambda x: (x is None, x)):
            group = buckets[key]
            for i in range(0, len(group), self.batch_size):
                b = group[i : i + self.batch_size]
                if not self.drop_last or len(b) == self.batch_size:
                    batches.append(b)
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]
        for b in batches:
            yield b
        self.seed += 1

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    out = {}
    for k in batch[0]:
        if isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch])
        else:
            out[k] = [b[k] for b in batch]
    return out


# ==========================
# LR + diagnostics
# ==========================
class CustomCurveLRScheduler:
    def __init__(self, optimizer, curve_points, max_train_steps):
        self.optimizer = optimizer
        self.curve_points = sorted(curve_points, key=lambda p: p[0])
        self.max_train_steps = max_train_steps
        self.current_training_step = 0
        if self.curve_points[0][0] != 0.0:
            self.curve_points.insert(0, [0.0, self.curve_points[0][1]])
        if self.curve_points[-1][0] != 1.0:
            self.curve_points.append([1.0, self.curve_points[-1][1]])
        self._update_lr()

    def _interp(self, x):
        x = max(0.0, min(1.0, x))
        for (x1, y1), (x2, y2) in zip(self.curve_points, self.curve_points[1:]):
            if x1 <= x <= x2:
                t = (x - x1) / (x2 - x1) if x2 != x1 else 0.0
                return y1 + t * (y2 - y1)
        return self.curve_points[-1][1]

    def _update_lr(self):
        x = self.current_training_step / max(1, self.max_train_steps - 1)
        lr = self._interp(x)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def step(self, training_step: int):
        self.current_training_step = training_step
        self._update_lr()


class TrainingDiagnostics:
    def __init__(self, accumulation_steps: int, test_param_name: str):
        self.accum = accumulation_steps
        self.test_param_name = test_param_name
        self.losses = deque(maxlen=accumulation_steps)

    def step(self, loss):
        self.losses.append(loss)

    def report(self, global_step, optimizer, raw_grad_norm, clipped_grad_norm, before_val, after_val):
        if not self.losses:
            return
        avg_loss = sum(self.losses) / len(self.losses)
        lr = optimizer.param_groups[0]["lr"]
        upd = torch.abs(after_val - before_val).max().item()
        reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
        alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        tqdm.write(
            f"\n--- Step {global_step} | Loss {avg_loss:.5f} | LR {lr:.2e} ---\n"
            f"  Grad (raw/clipped): {raw_grad_norm:.4f} / {clipped_grad_norm:.4f}\n"
            f"  VRAM: reserved={reserved:.2f}GB allocated={alloc:.2f}GB\n"
            f"  Param: {self.test_param_name} | update={upd:.3e}"
        )
        self.losses.clear()


# ==========================
# Optimizer factory
# ==========================
def create_optimizer(config: TrainingConfig, params):
    opt_type = config.OPTIMIZER_TYPE.lower()
    if opt_type == "raven":
        p = getattr(config, "RAVEN_PARAMS", default_config.RAVEN_PARAMS)
        return RavenAdamW(
            params,
            lr=config.LEARNING_RATE,
            betas=tuple(p.get("betas", [0.9, 0.999])),
            eps=p.get("eps", 1e-8),
            weight_decay=p.get("weight_decay", 0.01),
            debias_strength=p.get("debias_strength", 1.0),
        )
    if opt_type == "adafactor":
        p = getattr(config, "ADAFACTOR_PARAMS", default_config.ADAFACTOR_PARAMS)
        return Adafactor(
            params,
            lr=config.LEARNING_RATE,
            eps=tuple(p.get("eps", [1e-30, 1e-3])),
            clip_threshold=p.get("clip_threshold", 1.0),
            decay_rate=p.get("decay_rate", -0.8),
            beta1=p.get("beta1", None),
            weight_decay=p.get("weight_decay", 0.01),
            scale_parameter=p.get("scale_parameter", True),
            relative_step=p.get("relative_step", False),
            warmup_init=p.get("warmup_init", False),
        )
    raise ValueError(f"Unknown optimizer type: {opt_type}")


# ==========================
# Timestep sampler (top-level, no runtime deps)
# ==========================
class TimestepSampler:
    def __init__(self, config: TrainingConfig, noise_scheduler, device):
        self.config = config
        self.scheduler = noise_scheduler
        self.device = device
        self.num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        
        self.method = getattr(config, "TIMESTEP_SAMPLING_METHOD", "Random Integer")

        # Only enable LogSNR logic if the sampling method actually uses it
        self.use_log_snr = getattr(config, "USE_LOG_SNR", False) and ("LogSNR" in self.method)
        
        # ADD THIS: Disable LogSNR for flow matching
        is_flow = getattr(config, "PREDICTION_TYPE", "") == "flow_matching"
        if is_flow and self.use_log_snr:
            print("WARNING: LogSNR sampling not recommended for flow matching. Disabling.")
            self.use_log_snr = False
        
        self.log_snr_per_t = None

        print(f"\n{'=' * 60}")
        print("TIMESTEP SAMPLER INITIALIZATION")
        print(f"{'=' * 60}")
        print(f"Method: {self.method}")
        print(f"USE_LOG_SNR flag: {self.use_log_snr}")

        if self.use_log_snr:
            if hasattr(self.scheduler, "alphas_cumprod"):
                alphas_cumprod = self.scheduler.alphas_cumprod
                if not torch.is_tensor(alphas_cumprod):
                    alphas_cumprod = torch.tensor(
                        alphas_cumprod, device=device, dtype=torch.float32
                    )
                else:
                    alphas_cumprod = alphas_cumprod.to(
                        device=device, dtype=torch.float32
                    )

                # clamp to avoid log(0)
                alphas_cumprod = alphas_cumprod.clamp(1e-7, 1.0 - 1e-7)

                # log SNR = log(α² / (1-α²))
                self.log_snr_per_t = torch.log(
                    alphas_cumprod / (1.0 - alphas_cumprod)
                )
                self.log_snr_min = self.log_snr_per_t.min().item()
                self.log_snr_max = self.log_snr_per_t.max().item()

                print("[OK] LogSNR initialized successfully")
                print(f"  - SNR range: [{self.log_snr_min:.4f}, {self.log_snr_max:.4f}]")
                print(f"  - Timesteps: {len(self.log_snr_per_t)}")
            else:
                # For schedulers like FlowMatchEulerDiscreteScheduler
                print(
                    "[WARN] LogSNR sampling requested but scheduler has no "
                    "alphas_cumprod. Falling back to timestep space."
                )
        else:
            print("[INFO] LogSNR-based sampling disabled for this method")

        print(f"{'=' * 60}\n")

        # Logit-normal parameters (timestep-space)
        self.logit_normal_mean = float(getattr(config, "LOGIT_NORMAL_MEAN", 0.0))
        self.logit_normal_std = float(getattr(config, "LOGIT_NORMAL_STD", 1.0))

        # dynamic bounds
        self.current_min_ts = float(getattr(config, "TIMESTEP_SAMPLING_MIN", 0))
        self.current_max_ts = float(
            getattr(config, "TIMESTEP_SAMPLING_MAX", self.num_train_timesteps - 1)
        )
        self.global_min_ts = 0.0
        self.global_max_ts = float(self.num_train_timesteps - 1)
        self.target_min_grad = float(
            getattr(config, "TIMESTEP_SAMPLING_GRAD_MIN", 0.5)
        )
        self.target_max_grad = float(
            getattr(config, "TIMESTEP_SAMPLING_GRAD_MAX", 2.0)
        )
        self.adjustment_strength = 0.05
        self.smoothing_factor = 0.9
        self.max_shift_per_step = 50.0
        self.smoothed_grad_norm = (
            self.target_min_grad + self.target_max_grad
        ) / 2.0
        self.last_timestep_avg = self.num_train_timesteps / 2.0
        self.consecutive_spike_count = 0

    def sample(self, batch_size: int):
        # 1) LogSNR-based sampling (e.g. "Uniform LogSNR")
        if (
            self.use_log_snr
            and self.log_snr_per_t is not None
            and "LogSNR" in self.method
        ):
            if not hasattr(self, "_logsnr_sample_count"):
                self._logsnr_sample_count = 0
                print("\n>>> Using Uniform LogSNR sampling <<<\n")

            self._logsnr_sample_count += 1
            if self._logsnr_sample_count % 100 == 0:
                print(f"[LogSNR] Sampled {self._logsnr_sample_count} batches")

            u = torch.rand(batch_size, device=self.device)
            log_snr = self.log_snr_min + u * (self.log_snr_max - self.log_snr_min)

            diffs = (self.log_snr_per_t.view(1, -1) - log_snr.view(-1, 1)).abs()
            idx = diffs.argmin(dim=1).long()
            return idx

        # 2) Logit-normal over discrete timesteps (FlowMatch + Logit Normal)
        if "Logit Normal" in self.method:
            if not hasattr(self, "_logit_msg_shown"):
                self._logit_msg_shown = True
                # ASCII-only to avoid Windows cp1252 issues
                print(
                    f"\n>>> Using Logit Normal sampling "
                    f"(mu={self.logit_normal_mean}, sigma={self.logit_normal_std}) <<<\n"
                )

            # z ~ N(mu, sigma)
            z = (
                torch.randn(batch_size, device=self.device) * self.logit_normal_std
                + self.logit_normal_mean
            )
            # map to (0,1) via sigmoid
            u = torch.sigmoid(z)
            # scale to timestep range [0, num_train_timesteps-1]
            t = (u * (self.num_train_timesteps - 1)).long()
            t = torch.clamp(t, 0, self.num_train_timesteps - 1)
            return t

        # 3) Uniform continuous in timestep space
        if "Uniform Continuous" in self.method:
            if not hasattr(self, "_uniform_msg_shown"):
                self._uniform_msg_shown = True
                print("\n>>> Using Uniform Continuous sampling <<<\n")
            t = torch.rand(batch_size, device=self.device)
            return (t * (self.num_train_timesteps - 1)).long()

        # 4) Dynamic windowed sampling
        if "Dynamic" in self.method:
            if not hasattr(self, "_dynamic_msg_shown"):
                self._dynamic_msg_shown = True
                print("\n>>> Using Dynamic windowed sampling <<<\n")
            mn = int(self.current_min_ts)
            mx = int(self.current_max_ts)
            if mn >= mx:
                mn = max(0, mx - 1)
            return torch.randint(mn, mx + 1, (batch_size,), device=self.device)

        # 5) Fallback: plain random integer timesteps
        if not hasattr(self, "_fallback_msg_shown"):
            self._fallback_msg_shown = True
            print("\n>>> Using Random Integer sampling (fallback) <<<\n")
        mn = int(getattr(self.config, "TIMESTEP_SAMPLING_MIN", 0))
        mx = int(
            getattr(
                self.config,
                "TIMESTEP_SAMPLING_MAX",
                self.num_train_timesteps - 1,
            )
        )
        return torch.randint(mn, mx + 1, (batch_size,), device=self.device)

    def update(self, raw_grad_norm: float):
        if "Dynamic" not in self.method:
            return
        # spike detection
        spike = raw_grad_norm > self.target_max_grad
        if spike:
            self.consecutive_spike_count += 1
        else:
            self.consecutive_spike_count = 0
        # EMA
        self.smoothed_grad_norm = (
            self.smoothing_factor * self.smoothed_grad_norm
            + (1 - self.smoothing_factor) * raw_grad_norm
        )
        base_shift = self.adjustment_strength * self.num_train_timesteps
        shift_dir = 0.0
        if self.consecutive_spike_count > 1:
            shift_dir = -1.5
        elif self.consecutive_spike_count == 1:
            shift_dir = -0.5
        else:
            if self.smoothed_grad_norm < self.target_min_grad:
                shift_dir = 1.0
            elif self.smoothed_grad_norm > self.target_max_grad:
                shift_dir = -1.0
        if shift_dir != 0.0:
            delta = min(base_shift * abs(shift_dir), self.max_shift_per_step)
            delta = delta * (shift_dir / abs(shift_dir))
            self.current_min_ts += delta
            self.current_max_ts += delta
        # clamp + min width
        self.current_min_ts = max(self.global_min_ts, self.current_min_ts)
        self.current_max_ts = min(self.global_max_ts, self.current_max_ts)
        min_width = 50.0
        if (self.current_max_ts - self.current_min_ts) < min_width:
            c = (self.current_max_ts + self.current_min_ts) / 2.0
            self.current_min_ts = c - min_width / 2.0
            self.current_max_ts = c + min_width / 2.0
        self.current_min_ts = max(self.global_min_ts, self.current_min_ts)
        self.current_max_ts = min(self.global_max_ts, self.current_max_ts)

    def record_timesteps(self, timesteps: torch.Tensor):
        if "Dynamic" in self.method:
            self.last_timestep_avg = timesteps.float().mean().item()

# lil patch

def patch_diffusers_single_file():
    """Globally force diffusers to accept pathlib.Path on Windows."""
    try:
        from diffusers import StableDiffusionXLPipeline
    except Exception:
        return
    orig = StableDiffusionXLPipeline.from_single_file

    def _patched(path, *args, **kwargs):
        from pathlib import Path
        if isinstance(path, Path):
            path = str(path)
        return orig(path, *args, **kwargs)

    StableDiffusionXLPipeline.from_single_file = _patched

# ==========================
# Main training entry
# ==========================
def main():
    patch_diffusers_single_file()
    config = TrainingConfig()
    if config.SEED:
        set_seed(config.SEED)

    out_dir = Path(config.OUTPUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) caching pass if needed
    if check_if_caching_needed(config):
        print("INFO: Caching required. Loading VAE + text encoders...")
        vae = load_vae_only(config, device)
        if vae is None:
            pipe = StableDiffusionXLPipeline.from_single_file(
                normalize_model_path(config.SINGLE_FILE_CHECKPOINT_PATH),
                torch_dtype=torch.float32,
                device_map=None,
            )
            tokenizer = pipe.tokenizer; tokenizer_2 = pipe.tokenizer_2
            te1 = pipe.text_encoder; te2 = pipe.text_encoder_2
            vae = pipe.vae.to(device); del pipe; gc.collect()
        else:
            pipe = StableDiffusionXLPipeline.from_single_file(
                normalize_model_path(config.SINGLE_FILE_CHECKPOINT_PATH),
                torch_dtype=config.compute_dtype,
            )
            tokenizer = pipe.tokenizer; tokenizer_2 = pipe.tokenizer_2
            te1 = pipe.text_encoder; te2 = pipe.text_encoder_2
            del pipe; gc.collect(); torch.cuda.empty_cache()
        precompute_and_cache_latents(config, tokenizer, tokenizer_2, te1, te2, vae, device)
        del tokenizer, tokenizer_2, te1, te2, vae; gc.collect(); torch.cuda.empty_cache()
    else:
        print("INFO: All datasets already cached. Skipping caching.")

    # 2) load pipeline for training
    model_path = get_training_model_path(config)
    print(f"Loading training UNet from {model_path}")
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=config.compute_dtype,
    )
    orig_sched_cfg = pipe.scheduler.config
    unet = pipe.unet
    del pipe; gc.collect(); torch.cuda.empty_cache()

    base_model_sd = load_file(model_path)

    # 3) scheduler for training
    SCHED_MAP = {
        "DDPMScheduler": DDPMScheduler,
        "DDIMScheduler": DDIMScheduler,
        "EulerDiscreteScheduler": EulerDiscreteScheduler,
        "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    }
    
    sched_name = getattr(config, "NOISE_SCHEDULER", "DDPMScheduler").replace(" (Experimental)", "")
    is_flow_matching = config.PREDICTION_TYPE == "flow_matching"
    
    # Validate scheduler/prediction_type compatibility
    if is_flow_matching:
        if sched_name != "FlowMatchEulerDiscreteScheduler":
            print(f"WARNING: Flow matching requires FlowMatchEulerDiscreteScheduler. Overriding {sched_name}.")
            sched_name = "FlowMatchEulerDiscreteScheduler"
    else:
        if sched_name == "FlowMatchEulerDiscreteScheduler":
            print(f"WARNING: FlowMatchEulerDiscreteScheduler requires flow_matching. Switching to DDPMScheduler.")
            sched_name = "DDPMScheduler"
    
    sched_cls = SCHED_MAP.get(sched_name)
    if not sched_cls:
        raise ValueError(f"Unknown scheduler: {sched_name}")
    
    # Build scheduler config
    train_sched_cfg = dict(orig_sched_cfg)
    
    if is_flow_matching:
        # FlowMatch-specific settings
        train_sched_cfg["shift"] = getattr(config, "FLOW_MATCHING_SHIFT", 3.0)
        train_sched_cfg["num_train_timesteps"] = 1000
        # Remove incompatible standard diffusion keys
        for key in ["prediction_type", "beta_schedule", "beta_start", "beta_end", "trained_betas"]:
            train_sched_cfg.pop(key, None)
        print(f"INFO: Flow matching enabled (shift={train_sched_cfg['shift']})")
    else:
        train_sched_cfg["prediction_type"] = config.PREDICTION_TYPE
    
    train_sched_cfg = filter_scheduler_config(train_sched_cfg, sched_cls)
    noise_scheduler = sched_cls.from_config(train_sched_cfg)
    
    print(f"INFO: Scheduler={sched_cls.__name__}, Prediction={config.PREDICTION_TYPE}")
    
    if is_flow_matching:
        print(f"      Flow matching.")
    else:
        print(f"      Prediction type: {noise_scheduler.config.prediction_type}")

    # 4) optional zero-terminal SNR
    if getattr(config, "USE_ZERO_TERMINAL_SNR", False) and hasattr(noise_scheduler, "betas"):
        betas = rescale_zero_terminal_snr(noise_scheduler.betas.to(torch.float32))
        noise_scheduler.betas = betas
        noise_scheduler.alphas = 1.0 - betas
        noise_scheduler.alphas_cumprod = torch.cumprod(noise_scheduler.alphas, dim=0)
        noise_scheduler.num_train_timesteps = betas.shape[0]
        print("INFO: Applied zero-terminal SNR")

    # 5) move unet to device + attention
    attn_mode = getattr(config, "MEMORY_EFFICIENT_ATTENTION", "").lower()
    if attn_mode == "xformers":
        try:
            unet.enable_xformers_memory_efficient_attention(); print("INFO: xFormers enabled")
        except Exception as e:
            print(f"WARNING: could not enable xFormers: {e}")
    elif attn_mode == "sdpa":
        unet.set_attn_processor(AttnProcessor2_0()); print("INFO: SDPA enabled")
    unet.to(device); unet.enable_gradient_checkpointing()

    # --- Apply reflection padding if enabled ---
    if getattr(config, "USE_REFLECTION_PADDING", False):
        apply_reflection_padding_to_unet(unet, enabled=True)

    # 6) sampler instance
    timestep_sampler = TimestepSampler(config, noise_scheduler, device)
    print(f"--- Using Timestep Sampling: {timestep_sampler.method} ---")

    # 7) param selection / LoRA injection
    use_lora = getattr(config, "USE_LORA", False)

    if use_lora:
        adapter_type = getattr(config, "LORA_TYPE", "LoRA")
        
        print("=" * 40)
        print(f"{adapter_type.upper()} MODE ENABLED (PEFT)")
        print("=" * 40)

        if adapter_type == "LoRA":
            unet = inject_lora_peft(
                unet,
                config,
                target_modules=getattr(config, "LORA_TARGET_MODULES", None),
            )
        elif adapter_type == "LoCon":
            unet = inject_locon_peft(
                unet,
                config,
                target_modules=getattr(config, "LOCON_TARGET_MODULES", None),
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

        params_to_opt = [p for p in unet.parameters() if p.requires_grad]
        trainable_names = [n for n, p in unet.named_parameters() if p.requires_grad]
        frozen_names = [n for n, p in unet.named_parameters() if not p.requires_grad]

        train_params = sum(p.numel() for p in params_to_opt)
        total_params = sum(p.numel() for p in unet.parameters())

        print(f"Total UNet params: {total_params / 1e6:.2f}M")
        print(f"Trainable {adapter_type} params: {train_params / 1e6:.2f}M ({train_params/total_params*100:.4f}%)")

    else:
        # Original full-model training
        trainable_names, frozen_names = [], []
        for name, param in unet.named_parameters():
            param.requires_grad = True
            if any(ex in name for ex in config.UNET_EXCLUDE_TARGETS):
                param.requires_grad = False
                frozen_names.append(name)
            else:
                trainable_names.append(name)
        
        params_to_opt = [p for p in unet.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in unet.parameters())
        train_params = sum(p.numel() for p in params_to_opt)
        
        print(f"Total UNet params: {total_params / 1e6:.2f}M")
        print(f"Trainable params: {train_params / 1e6:.2f}M ({train_params/total_params*100:.2f}%)")

    unet.to(device)  # Ensure UNet is on the correct device after LoRA injection

    if not params_to_opt:
        raise ValueError("No parameters selected for training.")

    # 8) optimizer + LR
    optimizer = create_optimizer(config, params_to_opt)
    lr_curve = getattr(
        config,
        "LR_CUSTOM_CURVE",
        [[0.0, 0.0], [0.1, config.LEARNING_RATE], [1.0, 0.0]],
    )
    lr_scheduler = CustomCurveLRScheduler(optimizer, lr_curve, config.MAX_TRAIN_STEPS)

    # 9) dataset + loader
    dataset = ImageTextLatentDataset(config)
    batch_sampler = BucketBatchSampler(dataset, config.BATCH_SIZE, config.SEED, True, True)
    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=custom_collate_fn,
        num_workers=config.NUM_WORKERS,
    )

    # 10) resume
    global_step = 0
    state_path = Path(config.RESUME_STATE_PATH)
    if config.RESUME_TRAINING and state_path.exists():
        st = torch.load(state_path, map_location="cpu")
        optimizer.load_state_dict(st["optimizer_state_dict"])
        global_step = st["step"] * config.GRADIENT_ACCUMULATION_STEPS
        print(f"Resumed optimizer from {state_path}, global_step={global_step}")

    is_fp32 = next(unet.parameters()).dtype == torch.float32
    use_scaler = is_fp32 and config.MIXED_PRECISION in ["float16", "fp16"]
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    is_v_pred = config.PREDICTION_TYPE == "v_prediction"

    if use_lora:
        # For LoRA, watch the first trainable (LoRA) parameter
        watch_name = trainable_names[0]
        test_param = dict(unet.named_parameters())[watch_name]
    else:
        watch_name = trainable_names[0]
        test_param = dict(unet.named_parameters())[watch_name]
    
    diagnostics = TrainingDiagnostics(config.GRADIENT_ACCUMULATION_STEPS, watch_name)

    print("=" * 40)
    print("NOISE SETTINGS")
    if getattr(config, "USE_NOISE_OFFSET", False):
        print(f"Noise offset enabled, strength={config.NOISE_OFFSET}")
    else:
        print("Noise offset disabled")
    print("=" * 40)
    
    # VALIDATION BLOCK:
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION VALIDATION")
    print("="*60)
    print(f"Scheduler: {type(noise_scheduler).__name__}")
    print(f"Prediction type: {config.PREDICTION_TYPE}")
    print(f"Flow matching: {is_flow_matching}")
    print(f"VAE scaling: {config.VAE_SCALING_FACTOR}")
    print(f"VAE shift: {config.VAE_SHIFT_FACTOR}")
    print(f"Timestep sampling: {timestep_sampler.method}")
    print(f"Num train timesteps: {noise_scheduler.config.num_train_timesteps}")
    if is_flow_matching:
        print(f"Flow shift: {noise_scheduler.config.shift}")
    print("="*60 + "\n")

    # 11) training loop
    unet.train()
    
    pbar = tqdm(range(global_step, config.MAX_TRAIN_STEPS), desc="Training", total=config.MAX_TRAIN_STEPS, initial=global_step)
    accumulated_paths = []
    done = False
    while not done:
        for batch in dataloader:
            if global_step >= config.MAX_TRAIN_STEPS:
                done = True; break
            if not batch:
                continue

            if "latent_path" in batch:
                accumulated_paths.extend(batch["latent_path"])

            latents = batch["latents"].to(device, non_blocking=True)
            embeds = batch["embeds"].to(device, non_blocking=True)
            pooled = batch["pooled"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=config.compute_dtype, enabled=True):
                time_ids = torch.cat(
                    [
                        torch.tensor(list(osz) + [0, 0] + list(tsz)).unsqueeze(0)
                        for osz, tsz in zip(batch["original_sizes"], batch["target_sizes"])
                    ],
                    dim=0,
                ).to(device, dtype=embeds.dtype)

                if getattr(config, "USE_NOISE_OFFSET", False):
                    noise = generate_offset_noise(latents, config)
                else:
                    noise = torch.randn_like(latents)

                timesteps = timestep_sampler.sample(latents.shape[0])
                timestep_sampler.record_timesteps(timesteps)

                is_flow_matching = config.PREDICTION_TYPE == "flow_matching"

                if is_flow_matching:
                    # 1. Normalize t to [0, 1]
                    t = timesteps.float() / (noise_scheduler.config.num_train_timesteps - 1)
                    
                    # 2. Retrieve Shift value (SD3/Flux use 3.0)
                    shift_val = getattr(config, "FLOW_MATCHING_SHIFT", 3.0)
                    
                    # 3. Apply Time-Shifting Math
                    # Formula: t_shifted = (t * shift) / (1 + (shift - 1) * t)
                    if shift_val != 1.0:
                        t = (t * shift_val) / (1 + (shift_val - 1) * t)

                    # 4. Interpolate x_t = (1-t)*x_0 + t*x_1
                    t_expanded = t.view(-1, 1, 1, 1)
                    noisy_latents = (1.0 - t_expanded) * latents + t_expanded * noise
                    
                    # 5. Target is the velocity field: v = x_1 - x_0
                    target = noise - latents
                else:
                    # Standard diffusion
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    if is_v_pred:
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                pred = unet(
                    noisy_latents,
                    timesteps,
                    embeds,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
                ).sample

                loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

            diagnostics.step(loss.item())
            scaler.scale(loss / config.GRADIENT_ACCUMULATION_STEPS).backward()

            if (global_step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                current_optim_step = (global_step + 1) // config.GRADIENT_ACCUMULATION_STEPS
                before = test_param.data.clone()
                scaler.unscale_(optimizer)
                
                raw_norm = 0.0
                for p in params_to_opt:
                    if p.grad is not None:
                        g = p.grad.data.norm(2).item()
                        raw_norm += g * g
                raw_norm = raw_norm ** 0.5

                if config.CLIP_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(params_to_opt, config.CLIP_GRAD_NORM)
                    clipped = min(raw_norm, config.CLIP_GRAD_NORM)
                else:
                    clipped = raw_norm

                timestep_sampler.update(raw_norm)

                is_grad_anomaly = (
                    raw_norm > config.GRAD_SPIKE_THRESHOLD_HIGH
                    or raw_norm < config.GRAD_SPIKE_THRESHOLD_LOW
                )

                # Only log anomalies when *not* in LoRA mode
                if is_grad_anomaly and not use_lora:
                    tqdm.write("\n=== GRADIENT ANOMALY DETECTED ===")
                    tqdm.write(f"Step: {current_optim_step}")
                    tqdm.write(f"Raw Grad Norm: {raw_norm:.4f}")
                    tqdm.write(f"Clipped Grad Norm: {clipped:.4f}")
                    for pth in accumulated_paths:
                        tqdm.write(f"  - {Path(pth).stem}")
                    tqdm.write("===============================\n")


                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step(global_step + 1)
                optimizer.zero_grad(set_to_none=True)
                after = test_param.data.clone()

                avg_loss = sum(diagnostics.losses) / len(diagnostics.losses) if diagnostics.losses else 0
                pbar.set_postfix(loss=f"{avg_loss:.4f}")

                diagnostics.report(current_optim_step, optimizer, raw_norm, clipped, before, after)

                # Check optimizer step count, not global_step
                if current_optim_step % config.SAVE_EVERY_N_STEPS == 0:
                    ckpt_name = f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_step{current_optim_step}"
                    
                    torch.save(
                        {"step": current_optim_step, "optimizer_state_dict": optimizer.state_dict()},
                        ckpt_dir / f"{ckpt_name}_state.pt",
                    )
                    
                    if use_lora:
                        adapter_type = getattr(config, "LORA_TYPE", "LoRA")
                        
                        if adapter_type == "LoRA":
                            lora_state = extract_lora_state_dict_comfy_peft(unet)
                            pt_path = ckpt_dir / f"{ckpt_name}_lora.pt"
                        elif adapter_type == "LoCon":
                            lora_state = extract_locon_state_dict_comfy_peft(unet)
                            pt_path = ckpt_dir / f"{ckpt_name}_locon.pt"
                        
                        torch.save(lora_state, pt_path)
                        print(f"\nSaved {adapter_type} checkpoint at step {current_optim_step}")
                        
                        convert_lora_pt_to_safetensors(pt_path)
                    else:
                        # Save full model
                        ckpt_sd = base_model_sd.copy()
                        unet_sd = unet.state_dict()
                        key_map = _generate_hf_to_sd_unet_key_mapping(list(unet_sd.keys()))
                        for name in trainable_names:
                            mapped = key_map.get(name)
                            if mapped:
                                sd_key = "model.diffusion_model." + mapped
                                if sd_key in ckpt_sd:
                                    ckpt_sd[sd_key] = unet_sd[name].to(config.compute_dtype)
                        
                        save_file(ckpt_sd, ckpt_dir / f"{ckpt_name}.safetensors")
                        print(f"\nSaved checkpoint at step {current_optim_step}")
                
                accumulated_paths.clear()

            global_step += 1
            pbar.update(1)

    pbar.close()
    print("\nTraining complete.")

    # 12) final save
    final_optim_step = global_step // config.GRADIENT_ACCUMULATION_STEPS
    base_name = f"{Path(config.SINGLE_FILE_CHECKPOINT_PATH).stem}_final"
    print("Saving final model and state...")
    torch.save(
        {"step": final_optim_step, "optimizer_state_dict": optimizer.state_dict()},
        out_dir / f"{base_name}_state.pt",
    )

    if use_lora:
        adapter_type = getattr(config, "LORA_TYPE", "LoRA")
        
        if adapter_type == "LoRA":
            adapter_state = extract_lora_state_dict_comfy_peft(unet)
            final_path = out_dir / f"{base_name}_lora.pt"
        elif adapter_type == "LoCon":
            adapter_state = extract_locon_state_dict_comfy_peft(unet)
            final_path = out_dir / f"{base_name}_locon.pt"
        
        torch.save(adapter_state, final_path)
        print(f"Final {adapter_type} saved to: {final_path}")

        st_path = convert_lora_pt_to_safetensors(final_path)
        if st_path:
            print(f"Final {adapter_type} (safetensors) saved to: {st_path}")
    else:
        watch_name = trainable_names[0]
        test_param = dict(unet.named_parameters())[watch_name]

        # Save full model
        final_path = out_dir / f"{base_name}.safetensors"
        final_sd = base_model_sd.copy()
        unet_sd = unet.state_dict()
        key_map = _generate_hf_to_sd_unet_key_mapping(list(unet_sd.keys()))

        for name in trainable_names:
            mapped = key_map.get(name)
            if not mapped:
                continue
            sd_key = "model.diffusion_model." + mapped
            if sd_key in final_sd:
                final_sd[sd_key] = unet_sd[name].to(config.compute_dtype)

        save_file(final_sd, final_path)
        print(f"Final model saved to: {final_path}")

# ==========================
# Key mapping helper kept at end
# ==========================
def _generate_hf_to_sd_unet_key_mapping(hf_keys):
    mapping = {}
    for hf_key in hf_keys:
        key = hf_key
        if "resnets" in key:
            new_key = re.sub(
                r"^down_blocks\.(\d+)\.resnets\.(\d+)\.",
                lambda m: f"input_blocks.{3 * int(m.group(1)) + int(m.group(2)) + 1}.0.",
                key,
            )
            new_key = re.sub(
                r"^mid_block\.resnets\.(\d+)\.",
                lambda m: f"middle_block.{2 * int(m.group(1))}.",
                new_key,
            )
            new_key = re.sub(
                r"^up_blocks\.(\d+)\.resnets\.(\d+)\.",
                lambda m: f"output_blocks.{3 * int(m.group(1)) + int(m.group(2))}.0.",
                new_key,
            )
            new_key = (
                new_key.replace("norm1.", "in_layers.0.")
                .replace("conv1.", "in_layers.2.")
                .replace("norm2.", "out_layers.0.")
                .replace("conv2.", "out_layers.3.")
                .replace("time_emb_proj.", "emb_layers.1.")
                .replace("conv_shortcut.", "skip_connection.")
            )
            mapping[hf_key] = new_key
            continue
        if "attentions" in key:
            new_key = re.sub(
                r"^down_blocks\.(\d+)\.attentions\.(\d+)\.",
                lambda m: f"input_blocks.{3 * int(m.group(1)) + int(m.group(2)) + 1}.1.",
                key,
            )
            new_key = re.sub(r"^mid_block\.attentions\.0\.", "middle_block.1.", new_key)
            new_key = re.sub(
                r"^up_blocks\.(\d+)\.attentions\.(\d+)\.",
                lambda m: f"output_blocks.{3 * int(m.group(1)) + int(m.group(2))}.1.",
                new_key,
            )
            mapping[hf_key] = new_key
            continue
        if "downsamplers" in key:
            mapping[hf_key] = re.sub(
                r"^down_blocks\.(\d+)\.downsamplers\.0\.conv.",
                lambda m: f"input_blocks.{3 * (int(m.group(1)) + 1)}.0.op.",
                key,
            )
            continue
        if "upsamplers" in key:
            mapping[hf_key] = re.sub(
                r"^up_blocks\.(\d+)\.upsamplers\.0.",
                lambda m: f"output_blocks.{3 * int(m.group(1)) + 2}.2.",
                key,
            )
            continue
        if key.startswith("conv_in."):
            mapping[hf_key] = key.replace("conv_in.", "input_blocks.0.0.")
            continue
        if key.startswith("conv_norm_out."):
            mapping[hf_key] = key.replace("conv_norm_out.", "out.0.")
            continue
        if key.startswith("conv_out."):
            mapping[hf_key] = key.replace("conv_out.", "out.2.")
            continue
        if key.startswith("time_embedding.linear_1."):
            mapping[hf_key] = key.replace("time_embedding.linear_1.", "time_embed.0.")
            continue
        if key.startswith("time_embedding.linear_2."):
            mapping[hf_key] = key.replace("time_embedding.linear_2.", "time_embed.2.")
            continue
        if key.startswith("add_embedding.linear_1."):
            mapping[hf_key] = key.replace("add_embedding.linear_1.", "label_emb.0.0.")
            continue
        if key.startswith("add_embedding.linear_2."):
            mapping[hf_key] = key.replace("add_embedding.linear_2.", "label_emb.0.2.")
            continue
    return mapping


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

