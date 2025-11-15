This is a UNet trainer for SDXL models that requires atleast 12 GB of VRAM.    

Modified from [Aozora](https://github.com/Hysocs/Aozora_SDXL_Training) for personal usage.    

Features:    

- LoRA Mode (minimal toggle that freezes UNet layers).    
- More timestep sampling choices.
- Flow matching with FlowMatchEulerDiscrete scheduler.
- RavenAdamW buffers stored in fp32.  
- Optimizer state offload frequency.
- Optional reflection padding for EQ-VAE.    
- Neutral theme with squared borders.
- Bonus utility scripts.    

Usage:

`python -m venv venv`        
`./venv/scripts/activate`    
`pip install -r requirements.txt`      
`python gui.py`    

Example configs are provided. Trainer will automatically generate a default one if there's none.  
