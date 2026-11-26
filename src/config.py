import sys
from pathlib import Path

# If running as PyInstaller frozen executable, resolve path to executable directory
if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_NUM_WORKERS = 0

# Held-out TEST split used to evaluate the global model after every federated round.
TEST_DIR = PROJECT_ROOT / "data" / "test"
if not TEST_DIR.exists():
    fallback = Path(r"C:\Users\Acer\Desktop\chest_xray\test")
    if fallback.exists():
        TEST_DIR = fallback

