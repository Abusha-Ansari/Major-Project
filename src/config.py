from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_SIZE = 224
DEFAULT_BATCH_SIZE = 16
DEFAULT_LR = 1e-4
DEFAULT_NUM_WORKERS = 0

# Path to the held-out TEST split used to evaluate the global model after
# every federated round.  Must contain NORMAL/ and PNEUMONIA/ subfolders.
# This is the Kaggle chest-xray test split — NOT the training data.
# Set to None to disable automatic post-aggregation evaluation.
TEST_DIR = Path(r"C:\Users\Acer\Desktop\chest_xray\test")
