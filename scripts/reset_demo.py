"""
Demo Environment Reset Script
──────────────────────────────
Completely wipes all previous training history, aggregated models, registered hospitals,
client caches, metrics, and logs so that you can start a 100% fresh demonstration from scratch.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def clean_directory_contents(target_dir: Path, name: str):
    if target_dir.exists():
        for item in target_dir.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                print(f"  [!] Could not delete {item.name}: {e}")
        print(f"  [OK] Cleared all files in: {name} ({target_dir.name}/)")
    else:
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] Created empty directory: {name} ({target_dir.name}/)")


def main():
    parser = argparse.ArgumentParser(description="Reset Federated Learning project to clean demo state.")
    parser.add_argument("--train-seed", action="store_true", help="Immediately train fresh seed model after reset.")
    args = parser.parse_args()

    print("=" * 65)
    print("  FEDERATED LEARNING DEMO RESET UTILITY")
    print("=" * 65)
    print("Resetting all project storage, models, and hospital registries...\n")

    # 1. Clear models directory
    models_dir = PROJECT_ROOT / "models"
    clean_directory_contents(models_dir, "Global Models")

    # 2. Clear server storage (hospitals.json, round_state.json, metrics.json, fl_updates/)
    server_storage = PROJECT_ROOT / "server_storage"
    clean_directory_contents(server_storage, "Server Storage & Hospital Registry")

    # 3. Clear client work directory (downloaded models & local weights)
    client_work = PROJECT_ROOT / "client_work"
    clean_directory_contents(client_work, "Client Work Cache")

    # 4. Remove debug and runtime log files
    for log_file in PROJECT_ROOT.glob("*.log"):
        try:
            log_file.unlink()
            print(f"  [OK] Deleted log file: {log_file.name}")
        except Exception as e:
            print(f"  [!] Could not delete {log_file.name}: {e}")

    print("\n" + "-" * 65)
    print("  All past demo data and registered hospitals have been CLEARED!")
    print("-" * 65)

    if args.train_seed:
        print("\n[Optional] Training fresh initial baseline model on data/seed_train...")
        import subprocess
        venv_py = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
        py_cmd = str(venv_py) if venv_py.exists() else sys.executable
        train_script = PROJECT_ROOT / "src" / "train_initial.py"
        cmd = [
            py_cmd,
            str(train_script),
            "--train-dir", str(PROJECT_ROOT / "data" / "seed_train"),
            "--test-dir", str(PROJECT_ROOT / "data" / "test"),
            "--epochs", "2",
            "--output", str(models_dir / "xray_global.pt"),
        ]
        subprocess.check_call(cmd)
        print("\n  [OK] Fresh seed model trained and saved to models/xray_global.pt")
    else:
        print("\nReady for demonstration!")
        print("To train your initial model live during the demo, run:")
        print(r"  .\.venv\Scripts\python.exe src/train_initial.py --train-dir data/seed_train --test-dir data/test --epochs 2 --output models/xray_global.pt")
        print("\nThen launch the system with:")
        print("  .\\start.ps1")
        print("=" * 65)


if __name__ == "__main__":
    main()
