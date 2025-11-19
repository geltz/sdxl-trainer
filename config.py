# --- Paths ---
SINGLE_FILE_CHECKPOINT_PATH = "./model.safetensors"
VAE_PATH = ""  # Optional separate VAE path, leave empty to use VAE from model
VAE_SHIFT_FACTOR = 0.0  # Latent shift before scaling (0.0 for SDXL, non-zero for some RF models)
VAE_SCALING_FACTOR = 0.13025
USE_REFLECTION_PADDING = False  # Enable for EQ-VAE or better edge quality
OUTPUT_DIR = "./output"

# --- Resume Training ---
RESUME_TRAINING = False
RESUME_MODEL_PATH = ""
RESUME_STATE_PATH = ""

# --- Dataset Configuration ---
INSTANCE_DATASETS = [
    {
        "path": "./data",
        "repeats": 1,
    }
]

# --- Caching & Data Loaders ---
CACHING_BATCH_SIZE = 2
NUM_WORKERS = 0
SHOULD_UPSCALE = False
# --- Aspect Ratio Bucketing ---
TARGET_PIXEL_AREA = 1048576  # 1024*1024
MAX_AREA_TOLERANCE = 1.1

# --- Core Training Parameters ---
NOISE_SCHEDULER = "DDPMScheduler"  # Options: "DDPMScheduler", "DDIMScheduler", "EulerDiscreteScheduler", "FlowMatchEulerDiscreteScheduler"
PREDICTION_TYPE = "v_prediction"  # Options: "v_prediction", "epsilon", "flow_matching"
BETA_SCHEDULE = "scaled_linear"
MAX_TRAIN_STEPS = 6000
LEARNING_RATE = 3e-6
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
MIXED_PRECISION = "float16"
CLIP_GRAD_NORM = 1.0
SEED = 42

# --- Flow Matching Parameters ---
FLOW_MATCHING_SHIFT = 3.0

# --- Saving ---
SAVE_EVERY_N_STEPS = 100

# --- UNet Layer Exclusion (Blacklist) ---
UNET_EXCLUDE_TARGETS = "conv1, conv2"  # Comma-separated keywords to exclude from training

# --- Learning Rate Scheduler ---
LR_CUSTOM_CURVE = [
    [0.0, 0.0],
    [0.05, 8.0e-7],
    [0.85, 8.0e-7],
    [1.0, 1.0e-7]
]
LR_GRAPH_MIN = 0.0
LR_GRAPH_MAX = 1.0e-6

# --- Advanced ---
MEMORY_EFFICIENT_ATTENTION = "xformers"
USE_ZERO_TERMINAL_SNR = True

# --- Optimizer Configuration ---
OPTIMIZER_TYPE = "raven"  # "raven" or "adafactor"

# Raven Optimizer Parameters
RAVEN_PARAMS = {
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "weight_decay": 0.01,
    "debias_strength": 0.3,
    "use_grad_centralization": False,  # NEW
    "gc_alpha": 1.0  # NEW
}

# Adafactor Optimizer Parameters
ADAFACTOR_PARAMS = {
    "eps": [1e-30, 1e-3],
    "clip_threshold": 1.0,
    "decay_rate": -0.8,
    "beta1": None,
    "weight_decay": 0.01,
    "scale_parameter": True,
    "relative_step": False,
    "warmup_init": False
}

# --- Gradient Spike Detection ---
GRAD_SPIKE_THRESHOLD_HIGH = 75.0
GRAD_SPIKE_THRESHOLD_LOW = 0.2


USE_NOISE_OFFSET = False
NOISE_OFFSET = 0.0
USE_MULTISCALE_NOISE = False 

# --- Timestep Sampling ---
TIMESTEP_SAMPLING_MIN = 0
TIMESTEP_SAMPLING_MAX = 999
TIMESTEP_SAMPLING_GRAD_MIN = 0.5
TIMESTEP_SAMPLING_GRAD_MAX = 2.0

USE_LOG_SNR = True

TIMESTEP_SAMPLING_METHOD = "Uniform LogSNR"  # Options: "Dynamic", "Uniform Continuous", "Random Integer", "Uniform LogSNR", "Logit Normal"

# --- Logit Normal Sampling Parameters ---
LOGIT_NORMAL_MEAN = 0.0
LOGIT_NORMAL_STD = 1.0

# --- LoRA Configuration ---
USE_LORA = False
LORA_TYPE = "LoRA"

LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]

LOCON_RANK = 16
LOCON_ALPHA = 16
LOCON_DROPOUT = 0.0
LOCON_TARGET_MODULES = None

# --- Dataset Configuration ---
TAG_DROPOUT_RATE = 0.1  # 10% chance to drop each tag
TAG_DROPOUT_WHITELIST = "*girl*, *boy*, solo, masterpiece"  # Supports wildcards with *