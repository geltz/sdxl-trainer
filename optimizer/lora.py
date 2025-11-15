import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_up(self.dropout(self.lora_down(x))) * self.scaling

def inject_lora_into_unet(unet, rank: int, alpha: float, dropout: float, target_modules: List[str]):
    """Inject LoRA layers into UNet."""
    lora_layers = []
    
    # Collect target modules first
    targets = []
    for name, module in unet.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            targets.append((name, module))
    
    # Now modify them
    for name, module in targets:
        lora = LoRALinearLayer(
            module.in_features,
            module.out_features,
            rank,
            alpha,
            dropout
        )
        
        # Store original forward
        original_forward = module.forward
        
        # Create wrapper
        def make_forward(orig, lora_layer):
            def forward(x):
                return orig(x) + lora_layer(x)
            return forward
        
        module.forward = make_forward(original_forward, lora)
        module.lora = lora
        
        # Freeze original
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False
        
        lora_layers.append((name, lora))
    
    print(f"Injected LoRA into {len(lora_layers)} layers")
    return lora_layers

def extract_lora_state_dict(unet):
    """Extract only LoRA weights."""
    lora_state = {}
    for name, module in unet.named_modules():
        if hasattr(module, 'lora'):
            lora_state[f"{name}.lora_down.weight"] = module.lora.lora_down.weight
            lora_state[f"{name}.lora_up.weight"] = module.lora.lora_up.weight
    return lora_state