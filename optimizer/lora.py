import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class LoRAWrapper(nn.Module):
    def __init__(self, linear: nn.Linear, lora: LoRALinearLayer):
        super().__init__()
        self.linear = linear
        self.lora = lora
        
    def forward(self, x):
        return self.linear(x) + self.lora(x)

def inject_lora_into_unet(unet, rank: int, alpha: float, dropout: float, target_modules: List[str]):
    """Inject LoRA layers into UNet."""
    lora_layers = []
    
    def replace_module(model, name, new_module):
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)
    
    # Collect targets WITHOUT iterating named_modules
    targets = []
    for name, module in list(unet.named_modules()):
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            targets.append((name, module))
    
    # Replace modules
    for name, module in targets:
        lora = LoRALinearLayer(
            module.in_features,
            module.out_features,
            rank,
            alpha,
            dropout
        )
        
        # Freeze original
        module.weight.requires_grad = False
        if module.bias is not None:
            module.bias.requires_grad = False
        
        # Replace with wrapper
        wrapper = LoRAWrapper(module, lora)
        replace_module(unet, name, wrapper)
        
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