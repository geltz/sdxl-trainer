from safetensors.torch import load_file, save_file

src_path = "model1.safetensors"
dst_path = "model1_metadata.safetensors"

# 1. load all tensors
tensors = load_file(src_path)

# 2. build metadata according to Stability’s ModelSpec

# (enter your own values here)
metadata = {
    "modelspec.sai_model_spec": "1.0.0",              # version from the spec
    "modelspec.architecture": "stable-diffusion-xl-v1-base",  # or ...-refiner
    "modelspec.implementation": "sgm",                # or "diffusers", etc.
    "modelspec.title": "title",
    "modelspec.description": "description",
    "modelspec.author": "author",
    "modelspec.date": "yyyy-mm-dd",
    "modelspec.prediction_type": "v",
}

# 3. resave with metadata
save_file(tensors, dst_path, metadata=metadata)