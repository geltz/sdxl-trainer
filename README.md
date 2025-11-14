This can train SDXL checkpoints with 12 GB of VRAM as a strict minimum.
  
Note: This is UNet-only, the text encoders are left as-is.    

Modified from [Aozora](https://github.com/Hysocs/Aozora_SDXL_Training) for personal usage.    

Features:    

- Uniform LogSNR and Logit Normal timestep sampling.
- FlowMatchEulerDiscrete scheduler.
- RavenAdamW buffers stored in fp32.  
- Offload frequency value.
- Optional reflection padding for EQ-VAE.    
- Neutral theme with squared borders.  

Usage:

`python -m venv venv`        
`./venv/scripts/activate`    
`pip install -r requirements.txt`      
`python gui.py`    

An example config is provided. Trainer will automatically generate a default one if there's none.  
