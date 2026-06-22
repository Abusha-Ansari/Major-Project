import argparse
import random
import shutil
from pathlib import Path


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Split an ImageFolder dataset into simulated hospitals.")
    parser.add_argument("--source-dir", required=True, help="Source train directory with NORMAL/PNEUMONIA folders.")
    parser.add_argument("--output-dir", required=True, help="Output directory for hospital folders.")
    parser.add_argument("--num-hospitals", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def list_images(class_dir):
    return sorted(
        [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )


def main():
    args = parse_args()
    random.seed(args.seed)

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for hospital_idx in range(1, args.num_hospitals + 1):
        for class_name in CLASS_NAMES:
            (output_dir / f"hospital_{hospital_idx}" / class_name).mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES:
        images = list_images(source_dir / class_name)
        if not images:
            raise FileNotFoundError(f"No images found in {source_dir / class_name}")
        random.shuffle(images)

        for index, image_path in enumerate(images):
            hospital_idx = (index % args.num_hospitals) + 1
            target = output_dir / f"hospital_{hospital_idx}" / class_name / image_path.name
            shutil.copy2(image_path, target)

    for hospital_dir in sorted(output_dir.iterdir()):
        counts = {
            class_name: len(list_images(hospital_dir / class_name))
            for class_name in CLASS_NAMES
        }
        print(f"{hospital_dir.name}: {counts}")

    print(f"Created hospital splits at {output_dir}")


if __name__ == "__main__":
    main()
