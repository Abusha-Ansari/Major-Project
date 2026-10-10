"""
PyInstaller Build Script for Hospital Desktop Client
────────────────────────────────────────────────────
Compiles `src/hospital_desktop.py` into a portable, standalone folder `dist/HospitalClient/`
that can be copied to any USB drive and run on other laptops with ZERO installation.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
ENTRY_POINT = SRC_DIR / "hospital_desktop.py"


def main():
    print("=" * 65)
    print("  BUILDING STANDALONE HOSPITAL DESKTOP CLIENT (.EXE)")
    print("=" * 65)

    venv_pyinstaller = PROJECT_ROOT / ".venv" / "Scripts" / "pyinstaller.exe"
    pyinstaller_cmd = str(venv_pyinstaller) if venv_pyinstaller.exists() else "pyinstaller"

    cmd = [
        pyinstaller_cmd,
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name", "HospitalClient",
        "--paths", str(SRC_DIR),
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "torchvision.models",
        "--hidden-import", "PIL",
        "--hidden-import", "PIL.Image",
        "--hidden-import", "PIL.ImageTk",
        "--hidden-import", "requests",
        "--hidden-import", "matplotlib",
        "--hidden-import", "matplotlib.cm",
        "--hidden-import", "numpy",
        "--hidden-import", "tkinter",
        "--hidden-import", "tkinter.ttk",
        "--hidden-import", "tkinter.filedialog",
        "--hidden-import", "tkinter.messagebox",
        "--collect-all", "torch",
        "--collect-all", "torchvision",
        str(ENTRY_POINT),
    ]

    print(f"\n[1/3] Running PyInstaller command:")
    print(" ".join(cmd))
    print("\nCompiling (this may take 1-2 minutes for PyTorch)...")

    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))

    # Post-build: copy initial model and sample demo X-rays into the portable folder
    client_dist = DIST_DIR / "HospitalClient"
    print(f"\n[2/3] Setting up portable distribution assets in {client_dist}...")

    target_models = client_dist / "models"
    target_models.mkdir(parents=True, exist_ok=True)
    src_model = PROJECT_ROOT / "models" / "xray_global.pt"
    if src_model.exists():
        shutil.copy2(src_model, target_models / "xray_global.pt")
        print(f"  [OK] Bundled global model checkpoint: {target_models / 'xray_global.pt'}")

    target_samples = client_dist / "sample_xrays"
    target_samples.mkdir(parents=True, exist_ok=True)
    src_samples = PROJECT_ROOT / "data" / "sample_xrays"
    if src_samples.exists():
        for s in src_samples.glob("*.*"):
            shutil.copy2(s, target_samples / s.name)
        print(f"  [OK] Bundled sample X-rays for Grad-CAM XAI tab: {target_samples}")

    print("\n[3/3] Creating USB Ready Package folder structure...")
    usb_staging = PROJECT_ROOT / "USB_DEMO_PACKAGE"
    if usb_staging.exists():
        shutil.rmtree(usb_staging)
    usb_staging.mkdir(parents=True, exist_ok=True)

    # 1. Copy App
    print("  -> Copying HospitalClient app to USB_DEMO_PACKAGE...")
    shutil.copytree(client_dist, usb_staging / "HospitalClient")

    # 2. Copy Hospital data partitions
    src_hospitals = PROJECT_ROOT / "data" / "hospitals"
    if src_hospitals.exists():
        for h_dir in src_hospitals.iterdir():
            if h_dir.is_dir():
                print(f"  -> Copying {h_dir.name} dataset partition to USB_DEMO_PACKAGE...")
                shutil.copytree(h_dir, usb_staging / h_dir.name)

    # 3. Create a quick instructions text file on the USB
    readme_txt = usb_staging / "HOW_TO_RUN_DEMO.txt"
    readme_txt.write_text(
        """================================================================================
  FEDERATED LEARNING HOSPITAL NODE -- DEMONSTRATION GUIDE
================================================================================

Step 1: Connect this laptop to the same Wi-Fi network as the Admin laptop.

Step 2: Open the 'HospitalClient' folder and double-click 'HospitalClient.exe'.

Step 3: In the GUI:
  - Server URL:    http://<ADMIN_IP>:8000  (e.g., http://192.168.1.15:8000)
  - Hospital Name: Hospital_Alpha (or Hospital_Beta / Hospital_Gamma)
  - Password:      hospital123
  - Click 'Register' then 'Login'.

Step 4: Under 'Local Dataset & Training Setup':
  - Click 'Browse' on Dataset Folder.
  - Select the 'hospital_1' (or 'hospital_2') folder on this USB drive.
  - Click '⚡ Train Locally and Submit Weights'.

Step 5: Clinical Diagnostics & Grad-CAM XAI:
  - Click on the 'Clinical Diagnostics & Grad-CAM XAI' tab.
  - Browse for any sample X-ray in 'HospitalClient/sample_xrays/'.
  - Click '⚡ Run Diagnostic & Grad-CAM Analysis' to view the heatmap overlay!
================================================================================
""",
        encoding="utf-8",
    )
    print("  [OK] Created HOW_TO_RUN_DEMO.txt")

    print("\n" + "=" * 65)
    print("  BUILD SUCCESSFUL!")
    print(f"  - App Executable: {client_dist / 'HospitalClient.exe'}")
    print(f"  - Ready-to-copy USB Folder: {usb_staging}")
    print("  Simply copy the contents of 'USB_DEMO_PACKAGE' to your USB drive!")
    print("=" * 65)


if __name__ == "__main__":
    main()
