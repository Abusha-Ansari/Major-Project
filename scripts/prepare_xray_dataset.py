import argparse
import shutil
import zipfile
from pathlib import Path


EXPECTED_CLASSES = ["NORMAL", "PNEUMONIA"]


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and normalize the X-ray Kaggle dataset layout.")
    parser.add_argument("--zip-path", required=True, help="Path to the X-ray dataset ZIP file.")
    parser.add_argument("--output-dir", required=True, help="Output directory for normalized dataset.")
    return parser.parse_args()


def find_data_root(extract_dir):
    candidates = [
        extract_dir / "data",
        extract_dir,
    ]
    candidates.extend(path for path in extract_dir.rglob("data") if path.is_dir())

    for candidate in candidates:
        train_dir = candidate / "train"
        test_dir = candidate / "test"
        if train_dir.exists() and test_dir.exists():
            return candidate
    raise FileNotFoundError("Could not find data/train and data/test inside the ZIP.")


def copy_split(data_root, output_dir, split):
    source_split = data_root / split
    target_split = output_dir / split
    target_split.mkdir(parents=True, exist_ok=True)

    for class_name in EXPECTED_CLASSES:
        source_class = source_split / class_name
        if not source_class.exists():
            raise FileNotFoundError(f"Missing expected class folder: {source_class}")
        target_class = target_split / class_name
        if target_class.exists():
            shutil.rmtree(target_class)
        shutil.copytree(source_class, target_class)


def main():
    args = parse_args()
    zip_path = Path(args.zip_path)
    output_dir = Path(args.output_dir)
    temp_dir = output_dir.parent / "_xray_extract_tmp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(temp_dir)

    data_root = find_data_root(temp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_split(data_root, output_dir, "train")
    copy_split(data_root, output_dir, "test")

    shutil.rmtree(temp_dir)
    print(f"Prepared dataset at {output_dir}")


if __name__ == "__main__":
    main()
