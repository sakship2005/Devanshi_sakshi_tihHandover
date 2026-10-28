import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(class_dir):
    return sorted(
        p for p in class_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_tree_images(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    copied = 0

    for class_dir in sorted(p for p in src_dir.iterdir() if p.is_dir()):
        out_class_dir = dst_dir / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in list_images(class_dir):
            rel = image_path.relative_to(class_dir)
            out_path = out_class_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, out_path)
            copied += 1

    return copied


def split_external_domain(external_dir, output_dir, train_percent, val_percent, seed):
    external_dir = Path(external_dir)
    output_dir = Path(output_dir)
    rng = random.Random(seed)

    if train_percent < 0 or val_percent < 0:
        raise ValueError("Percent values must be non-negative.")
    if train_percent + val_percent >= 100:
        raise ValueError("train_percent + val_percent must be less than 100.")

    summary = {
        "external_dir": str(external_dir),
        "train_percent": train_percent,
        "val_percent": val_percent,
        "seed": seed,
        "classes": {},
    }

    for class_dir in sorted(p for p in external_dir.iterdir() if p.is_dir()):
        images = list_images(class_dir)
        rng.shuffle(images)

        total = len(images)
        train_count = int(round(total * train_percent / 100.0))
        val_count = int(round(total * val_percent / 100.0))

        target_train = images[:train_count]
        target_val = images[train_count:train_count + val_count]
        target_test = images[train_count + val_count:]

        for split_name, split_images in [
            ("target_train", target_train),
            ("target_val", target_val),
            ("target_test", target_test),
        ]:
            out_class_dir = output_dir / split_name / class_dir.name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            for image_path in split_images:
                shutil.copy2(image_path, out_class_dir / image_path.name)

        summary["classes"][class_dir.name] = {
            "total_external": total,
            "target_train": len(target_train),
            "target_val": len(target_val),
            "target_test": len(target_test),
        }

    return summary


def build_combined_training_set(source_processed_dir, output_dir):
    source_processed_dir = Path(source_processed_dir)
    output_dir = Path(output_dir)

    train_out = output_dir / "train"
    val_out = output_dir / "val"
    test_out = output_dir / "test"

    copied_source_train = copy_tree_images(source_processed_dir / "train", train_out)
    copied_source_val = copy_tree_images(source_processed_dir / "val", val_out)

    target_train_dir = output_dir / "target_train"
    target_val_dir = output_dir / "target_val"
    target_test_dir = output_dir / "target_test"

    copied_target_train = copy_tree_images(target_train_dir, train_out) if target_train_dir.exists() else 0
    copied_target_val = copy_tree_images(target_val_dir, val_out) if target_val_dir.exists() else 0
    copied_target_test = copy_tree_images(target_test_dir, test_out) if target_test_dir.exists() else 0

    return {
        "copied_source_train": copied_source_train,
        "copied_source_val": copied_source_val,
        "copied_target_train_into_train": copied_target_train,
        "copied_target_val_into_val": copied_target_val,
        "copied_remaining_target_into_test": copied_target_test,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a few-shot domain-transfer split from an external test/domain "
            "dataset and optionally combine it with the existing source train/val data."
        )
    )
    parser.add_argument("--source_processed_dir", type=str, default=None,
                        help="Existing processed source dataset containing train/val/test folders.")
    parser.add_argument("--external_dir", type=str, required=True,
                        help="External domain dataset with class subfolders.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for the new domain-transfer split.")
    parser.add_argument("--train_percent", type=float, required=True,
                        help="Percent of external images per class to add to training.")
    parser.add_argument("--val_percent", type=float, default=0.0,
                        help="Optional percent of external images per class to use for validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--combine_with_source", action="store_true",
                        help="Also create train, val, and test folders usable by train.py/evaluate.py.")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = split_external_domain(
        external_dir=args.external_dir,
        output_dir=output_dir,
        train_percent=args.train_percent,
        val_percent=args.val_percent,
        seed=args.seed,
    )

    if args.combine_with_source:
        if not args.source_processed_dir:
            raise ValueError("--source_processed_dir is required when --combine_with_source is used.")
        summary["combined_dataset"] = build_combined_training_set(
            source_processed_dir=args.source_processed_dir,
            output_dir=output_dir,
        )

    summary_path = output_dir / "domain_split_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Domain-transfer split created at: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
