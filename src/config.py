from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_NUM_WORKERS = 0
