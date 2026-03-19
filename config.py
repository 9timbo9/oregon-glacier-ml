# config.py
from pathlib import Path
import torch

# Base paths
BASE_DIR = Path(__file__).resolve().parent

# Years
YEARS = ['1980', '2000', '2020','2025']
CURRENT_YEAR_IDX =  3      # change to 0, 1, 2, or 3 as needed
CURRENT_YEAR = YEARS[CURRENT_YEAR_IDX]
LANDSATS = ['L5', 'L7', 'L8', 'L9']
LANDSAT = LANDSATS[CURRENT_YEAR_IDX]

# Data directories
DATA_DIR      = BASE_DIR / "data"
PATCHES_DIR   = BASE_DIR / "patches" / CURRENT_YEAR
MODELS_DIR    = BASE_DIR / "models"
OUTPUTS_DIR   = BASE_DIR / "outputs" / CURRENT_YEAR
OVERLAY_DIR   = OUTPUTS_DIR / "outline_images"
PATCH_COORDS_FILE = BASE_DIR / "patches" / "patch_coords.json"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
OVERLAY_DIR.mkdir(exist_ok=True)

# DEM file
DEM_PATH = DATA_DIR / "DEM" / "output_hh.tif"

# Model file (for inference)

MODEL_FILENAME = "glacier_unet_pseudolabel.pt"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# Feature flags
USE_SLOPE = True
USE_ASPECT = True

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters
BATCH_SIZE = 4
LR = 3e-4
RANDOM_CROP = 256               # training crop size (None = full image)
NDSI_THRESHOLD = 0.25           # threshold for pseudo-labels
MIN_OBJECT_PIXELS = 150         # remove small speckles in pseudo-labels
EPOCHS = 30

# Inference thresholds (optional, can be overridden in script)
PROB_THRESHOLD = 0.5            # base probability threshold
NDSI_THRESHOLD_INF = 0.35       # NDSI threshold for mask
ELEV_THRESHOLD = 2000   # was 300
MIN_PIXELS_CLEAN = 500           # min pixels after cleaning (≈0.05 km² at 30 m)