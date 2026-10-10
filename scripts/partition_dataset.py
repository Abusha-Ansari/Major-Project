"""
Federated Dataset Partitioning Script
──────────────────────────────────────
Splits an existing Chest X-Ray dataset into:
  1. `seed_train/`: A small seed dataset (e.g. 200 Normal + 200 Pneumonia) to train the initial Round 0 global model.
  2. `hospitals/hospital_1, hospital_2, hospital_3`: Non-overlapping local training partitions for client laptops.
  3. `test/`: Held-out evaluation split for post-aggregation global model validation.
  4. `sample_xrays/`: Curated demo X-rays for quick testing of the Grad-CAM XAI tab.
"""

import argparse
import random
import shutil
from pathlib import Path

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Partition X-ray dataset into Seed, Hospital Nodes, and Test sets.")
    parser.add_argument(
        "--source-train-dir",
        default=r"C:\Users\Acer\Desktop\chest_xray\train",
        help="Path to raw train folder containing NORMAL/ and PNEUMONIA/ subfolders.",
    )
    parser.add_argument(
        "--source-test-dir",
        default=r"C:\Users\Acer\Desktop\chest_xray\test",
        help="Path to raw test folder containing NORMAL/ and PNEUMONIA/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Base output directory.",
    )
    parser.add_argument(
        "--seed-samples-per-class",
        type=int,
        default=200,
        help="Number of images per class to reserve for initial Seed global model training.",
    )
    parser.add_argument(
        "--num-hospitals",
        type=int,
        default=3,
        help="Number of hospital node partitions.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def list_valid_images(class_dir: Path):
    if not class_dir.exists():
        return []
    return sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def main():
    args = parse_args()
    random.seed(args.random_seed)

    source_train = Path(args.source_train_dir)
    source_test = Path(args.source_test_dir)
    out_base = Path(args.output_dir)

    if not source_train.exists():
        raise FileNotFoundError(f"Source train directory does not exist: {source_train}")

    seed_dir = out_base / "seed_train"
    hospitals_dir = out_base / "hospitals"
    test_dir = out_base / "test"
    sample_dir = out_base / "sample_xrays"

    # Clean existing destination directories
    for d in [seed_dir, hospitals_dir, test_dir, sample_dir]:
        if d.exists():
            shutil.rmtree(d)

    # Initialize directories
    for cls in CLASS_NAMES:
        (seed_dir / cls).mkdir(parents=True, exist_ok=True)
        (test_dir / cls).mkdir(parents=True, exist_ok=True)
        for h_idx in range(1, args.num_hospitals + 1):
            (hospitals_dir / f"hospital_{h_idx}" / cls).mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  FEDERATED LEARNING DATASET PARTITIONER")
    print("=" * 65)

    # 1. Process Train Data -> Seed + Hospital Splits
    total_seed = 0
    hospital_counts = {f"hospital_{i}": {c: 0 for c in CLASS_NAMES} for i in range(1, args.num_hospitals + 1)}

    for cls in CLASS_NAMES:
        imgs = list_valid_images(source_train / cls)
        if not imgs:
            raise FileNotFoundError(f"No images found in {source_train / cls}")
        random.shuffle(imgs)

        # Allocate to Seed
        seed_count = min(args.seed_samples_per_class, len(imgs))
        seed_imgs = imgs[:seed_count]
        remaining_imgs = imgs[seed_count:]

        for img in seed_imgs:
            shutil.copy2(img, seed_dir / cls / img.name)
        total_seed += len(seed_imgs)

        # Distribute remaining to Hospital nodes
        for idx, img in enumerate(remaining_imgs):
            h_idx = (idx % args.num_hospitals) + 1
            h_name = f"hospital_{h_idx}"
            shutil.copy2(img, hospitals_dir / h_name / cls / img.name)
            hospital_counts[h_name][cls] += 1

    # 2. Copy Test Data
    total_test = 0
    if source_test.exists():
        for cls in CLASS_NAMES:
            test_imgs = list_valid_images(source_test / cls)
            for img in test_imgs:
                shutil.copy2(img, test_dir / cls / img.name)
            total_test += len(test_imgs)

    # 3. Create Sample Demo Images for XAI Tab
    for cls in CLASS_NAMES:
        test_imgs = list_valid_images(test_dir / cls)
        sample_picks = test_imgs[:3]
        for i, p in enumerate(sample_picks, 1):
            shutil.copy2(p, sample_dir / f"sample_{cls.lower()}_{i}{p.suffix}")

    # Summary Report
    print(f"\n[1] Seed Dataset (for Initial Global Model):")
    for cls in CLASS_NAMES:
        print(f"    - {cls}: {len(list_valid_images(seed_dir / cls))} images")
    print(f"    Total Seed Images: {total_seed} -> Saved in: {seed_dir}")

    print(f"\n[2] Hospital Node Partitions (for Laptop Nodes):")
    for h_name, counts in hospital_counts.items():
        total_h = sum(counts.values())
        print(f"    - {h_name}: {counts['NORMAL']} Normal, {counts['PNEUMONIA']} Pneumonia (Total: {total_h})")
    print(f"    Saved in: {hospitals_dir}")

    print(f"\n[3] Test Dataset (Held-out for Central Evaluation):")
    for cls in CLASS_NAMES:
        print(f"    - {cls}: {len(list_valid_images(test_dir / cls))} images")
    print(f"    Total Test Images: {total_test} -> Saved in: {test_dir}")

    print(f"\n[4] Curated Demo X-rays for Grad-CAM XAI tab:")
    sample_files = list(sample_dir.glob("*.*"))
    for sf in sample_files:
        print(f"    - {sf.name}")
    print(f"    Saved in: {sample_dir}")

    print("\n" + "=" * 65)
    print("  Dataset Partitioning Completed Successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
