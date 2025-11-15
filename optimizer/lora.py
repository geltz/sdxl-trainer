from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Optional PEFT import
# ---------------------------------------------------------------------------

try:
    from peft import LoraConfig, get_peft_model
except Exception:  # pragma: no cover - we just want a clean error message at runtime
    LoraConfig = None
    get_peft_model = None


# ---------------------------------------------------------------------------
# Legacy: manual LoRA implementation (kept for backwards compatibility)
# ---------------------------------------------------------------------------


class LoRALinearLayer(nn.Module):
    """Simple LoRA adapter for a Linear layer.

    This is the same structure as in the original file and is left here so
    existing training code that relies on it continues to work.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming init for A, zeros for B (common LoRA init)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.lora_up(self.dropout(self.lora_down(x))) * self.scaling


def inject_lora_into_unet(
    unet: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: List[str],
) -> List[tuple[str, LoRALinearLayer]]:
    """Legacy LoRA injector that directly wraps nn.Linear layers in the UNet.

    This version is unchanged from the original helper file so that older
    code keeps working. For new projects, prefer `inject_lora_peft`.
    """
    lora_layers: List[tuple[str, LoRALinearLayer]] = []

    device = next(unet.parameters()).device

    # Collect all target Linear modules first
    targets: List[tuple[str, nn.Linear]] = []
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            targets.append((name, module))

    for name, module in targets:
        lora = LoRALinearLayer(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        ).to(device)

        original_forward = module.forward

        def make_forward(orig, lora_layer):
            def forward(x):
                return orig(x) + lora_layer(x)
            return forward

        module.forward = make_forward(original_forward, lora)  # type: ignore[assignment]
        module.lora = lora  # type: ignore[attr-defined]
        module.lora_alpha = alpha  # type: ignore[attr-defined]
        module.lora_rank = rank  # type: ignore[attr-defined]

        # Freeze base weights
        module.weight.requires_grad_(False)
        if module.bias is not None:
            module.bias.requires_grad_(False)

        lora_layers.append((name, lora))

    print(f"[LoRA] Injected legacy LoRA into {len(lora_layers)} layers")
    return lora_layers


def extract_lora_state_dict(unet: nn.Module) -> Dict[str, torch.Tensor]:
    """Legacy extractor: returns a state_dict containing only the manually
    injected LoRA weights (lora_down / lora_up) keyed by module name.

    This is mainly useful for debugging or custom tooling. For exporting
    to ComfyUI, prefer `extract_lora_state_dict_comfy_peft`.
    """
    lora_state: Dict[str, torch.Tensor] = {}

    for name, module in unet.named_modules():
        if hasattr(module, "lora"):  # type: ignore[attr-defined]
            lora: LoRALinearLayer = module.lora  # type: ignore[assignment]
            key_prefix = name.replace(".", "_")

            lora_state[f"{key_prefix}.lora_down.weight"] = lora.lora_down.weight
            lora_state[f"{key_prefix}.lora_up.weight"] = lora.lora_up.weight

    return lora_state


def extract_lora_state_dict_comfy(unet: nn.Module) -> Dict[str, torch.Tensor]:
    """Legacy extractor for the manual LoRA implementation, exporting in a
    ComfyUI-compatible `lora_unet_*` format.

    NOTE: if you used PEFT (inject_lora_peft), call
          `extract_lora_state_dict_comfy_peft` instead.
    """
    lora_state: Dict[str, torch.Tensor] = {}

    for name, module in unet.named_modules():
        if hasattr(module, "lora"):  # type: ignore[attr-defined]
            lora: LoRALinearLayer = module.lora  # type: ignore[assignment]
            comfy_name = f"lora_unet_{name.replace('.', '_')}"

            lora_state[f"{comfy_name}.lora_down.weight"] = lora.lora_down.weight
            lora_state[f"{comfy_name}.lora_up.weight"] = lora.lora_up.weight

            # Optional alpha metadata; ComfyUI uses it as scaling factor if present
            alpha_val = getattr(module, "lora_alpha", getattr(lora, "alpha", None))
            if alpha_val is not None:
                lora_state[f"{comfy_name}.alpha"] = torch.tensor(float(alpha_val))

    return lora_state


# ---------------------------------------------------------------------------
# PEFT-based LoRA for UNet + ComfyUI export
# ---------------------------------------------------------------------------

_DEFAULT_TARGET_MODULES: List[str] = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "proj_in",
    "proj_out",
    "ff.net.0.proj",
    "ff.net.2",
]


@dataclass
class TrainingConfigLike:
    """Minimal config interface used by `inject_lora_peft`.

    Your actual training config object only needs to expose the attributes
    referenced here (extra attributes are ignored).
    """

    LORA_RANK: int
    LORA_ALPHA: int
    LORA_DROPOUT: float = 0.0


def _ensure_peft_available() -> None:
    if LoraConfig is None or get_peft_model is None:
        raise ImportError(
            "peft is not installed or failed to import. "
            "Install it with `pip install peft` before using inject_lora_peft."
        )


def inject_lora_peft(
    unet: nn.Module,
    config: TrainingConfigLike,
    target_modules: Optional[List[str]] = None,
    init_lora_weights: str | bool = "gaussian",
) -> nn.Module:
    """Wrap a diffusers UNet2DConditionModel with PEFT LoRA adapters.

    Parameters
    ----------
    unet:
        The UNet model instance (usually `UNet2DConditionModel`) from diffusers.
    config:
        Object exposing `LORA_RANK`, `LORA_ALPHA`, and `LORA_DROPOUT`.
    target_modules:
        Optional explicit list of module name fragments to transform. If ``None``,
        the standard Stable Diffusion set is used (to_q / to_k / to_v / to_out.0 /
        proj_in / proj_out / ff.net.0.proj / ff.net.2).
    init_lora_weights:
        Passed through to `LoraConfig`. For diffusion models `"gaussian"` is a
        good default (matches diffusers' own LoRA training scripts).

    Returns
    -------
    nn.Module
        The wrapped PEFT model. This is still drop-in compatible with the
        original UNet for training / inference, but exposes `peft_config` and
        LoRA parameters in its `state_dict()`.
    """
    _ensure_peft_available()

    if target_modules is None:
        target_modules = list(_DEFAULT_TARGET_MODULES)

    lora_cfg = LoraConfig(
        r=int(config.LORA_RANK),
        lora_alpha=int(config.LORA_ALPHA),
        lora_dropout=float(getattr(config, "LORA_DROPOUT", 0.0)),
        target_modules=target_modules,
        bias="none",
        init_lora_weights=init_lora_weights,
    )

    peft_unet = get_peft_model(unet, lora_cfg)

    # Keep a convenient handle to the LoRA config on the model itself
    if not hasattr(peft_unet, "lora_config_for_export"):
        peft_unet.lora_config_for_export = lora_cfg  # type: ignore[attr-defined]

    # Sanity check: make sure only LoRA parameters are trainable
    trainable = [n for n, p in peft_unet.named_parameters() if p.requires_grad]
    print(f"[LoRA-PEFT] Number of trainable parameters: {len(trainable)}")
    if trainable:
        print("[LoRA-PEFT] First few trainable parameter keys:")
        for n in trainable[:8]:
            print(f"  - {n}")

    return peft_unet


def _get_lora_alpha_from_peft_model(model: nn.Module) -> Optional[float]:
    """Try to recover a single `lora_alpha` value from a PEFT-wrapped model.

    We look in:
      * `model.lora_config_for_export` (set in `inject_lora_peft`), or
      * the first entry in `model.peft_config` if present.
    """
    # Preferred: explicit attribute we set during injection
    cfg = getattr(model, "lora_config_for_export", None)
    if cfg is not None and hasattr(cfg, "lora_alpha"):
        return float(cfg.lora_alpha)

    # Fallback: PEFT's own config dict
    peft_cfg = getattr(model, "peft_config", None)
    if isinstance(peft_cfg, Mapping) and peft_cfg:
        first = next(iter(peft_cfg.values()))
        if hasattr(first, "lora_alpha"):
            return float(first.lora_alpha)

    return None


def _iter_peft_lora_tensors(
    model: nn.Module,
) -> List[tuple[str, torch.Tensor]]:
    """Collect all LoRA A/B tensors from a PEFT-wrapped model.

    We look for parameters whose names contain `.lora_A` or `.lora_B`, which
    is how PEFT stores LoRA matrices for Linear layers.
    """
    out: List[tuple[str, torch.Tensor]] = []
    for key, value in model.state_dict().items():
        if ".lora_A" in key or ".lora_B" in key:
            out.append((key, value))
    return out


def extract_lora_state_dict_peft(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a plain state_dict containing only PEFT LoRA weights.

    Keys are left in PEFT's native format (e.g.
    `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight`).

    This is primarily useful for debugging / inspection; for ComfyUI export
    use `extract_lora_state_dict_comfy_peft` instead.
    """
    lora_tensors = _iter_peft_lora_tensors(model)
    return {k: v for k, v in lora_tensors}


def extract_lora_state_dict_comfy_peft(
    model: nn.Module,
    key_prefix: str = "lora_unet",
    to_cpu: bool = True,
) -> Dict[str, torch.Tensor]:
    """Extract ComfyUI-compatible LoRA from PEFT-wrapped UNet."""
    alpha = _get_lora_alpha_from_peft_model(model)
    lora_state: Dict[str, torch.Tensor] = {}
    
    # Map to track unique base keys to avoid duplicate alpha entries
    processed_bases = set()

    for full_key, tensor in _iter_peft_lora_tensors(model):
        # Determine if this is lora_A (down) or lora_B (up)
        if ".lora_A" in full_key:
            base_name, _, remainder = full_key.partition(".lora_A")
            kind = "down"
        elif ".lora_B" in full_key:
            base_name, _, remainder = full_key.partition(".lora_B")
            kind = "up"
        else:
            continue

        # Strip leading "base_model.model." or "unet." prefixes that PEFT adds
        if base_name.startswith("base_model.model."):
            base_name = base_name[len("base_model.model."):]
        elif base_name.startswith("unet."):
            base_name = base_name[len("unet."):]

        # ComfyUI key format: lora_unet_<module_path_with_underscores>
        comfy_base = f"{key_prefix}_{base_name.replace('.', '_')}"

        if to_cpu:
            tensor = tensor.detach().cpu()
        else:
            tensor = tensor.detach()

        # Add the weight tensors
        if kind == "down":
            lora_state[f"{comfy_base}.lora_down.weight"] = tensor
        elif kind == "up":
            lora_state[f"{comfy_base}.lora_up.weight"] = tensor

        # Add alpha once per base key
        if alpha is not None and comfy_base not in processed_bases:
            lora_state[f"{comfy_base}.alpha"] = torch.tensor(float(alpha))
            processed_bases.add(comfy_base)

    print(f"[LoRA Export] Exported {len(processed_bases)} LoRA layers for ComfyUI")
    
    # Debug: print first few keys
    sample_keys = list(lora_state.keys())[:6]
    if sample_keys:
        print("[LoRA Export] Sample keys:")
        for k in sample_keys:
            print(f"  {k}")
    
    return lora_state


# Convenience alias so older code can be updated with a minimal change:
# replace `extract_lora_state_dict_comfy` calls with this if you switch to PEFT.
extract_lora_state_dict_comfy_peft.__name__ = "extract_lora_state_dict_comfy_peft"
