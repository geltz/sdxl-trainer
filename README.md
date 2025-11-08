This can train SDXL checkpoints with 12 GB of VRAM as a strict minimum.

Modified from [Aozora](https://github.com/Hysocs/Aozora_SDXL_Training) for personal usage.    

Key changes:    

- Uniform LogSNR sampling for v-prediction.  
- RavenAdamW buffers stored in fp32.  
- Offload frequency value.  
- Neutral theme with squared borders.  

Usage:

`python -m venv venv`    
`./venv/scripts/activate`

(Before requirements, it is recommended to [install pytorch](https://pytorch.org/get-started/locally/) with proper libraries).    

`pip install -r requirements.txt`    
`python gui.py`  

Adjust settings to whatever. There is a config provided as an example.    
