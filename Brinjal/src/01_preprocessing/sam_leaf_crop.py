"""
sam_leaf_crop.py
Uses Segment Anything (automatic mask generation) to find the leaf as the
largest foreground object and crop tightly to it, removing background
(soil, blurred foliage, other leaves) that may be acting as a shortcut cue.

Run:
python sam_leaf_crop.py --input_dir Brinjal --output_dir Brinjal_leaf_cropped --checkpoint sam_vit_b.pth
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def load_sam(checkpoint_path, model_type='vit_b'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM ({model_type}) on {device}...")
    if device == "cpu":
        print("WARNING: CUDA not available — this will run on CPU and be very slow.")

    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,          # denser grid = better mask quality
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        min_mask_region_area=2000,   # ignore tiny noise masks (dirt, small spots)
    )
    return mask_generator


def get_leaf_bbox(image_rgb, mask_generator, padding_frac=0.05):
    """
    Runs SAM automatic mask generation and picks the largest mask that is
    NOT the whole image (to avoid picking the entire frame as one mask).
    Returns (x0, y0, x1, y1) bounding box, or None if no suitable mask found.
    """
    h, w = image_rgb.shape[:2]
    image_area = h * w

    masks = mask_generator.generate(image_rgb)
    if not masks:
        return None

    # Filter out masks that cover almost the entire image (likely background/frame)
    candidate_masks = [m for m in masks if m['area'] < 0.97 * image_area]
    if not candidate_masks:
        candidate_masks = masks  # fallback

    # Pick the largest remaining mask -> assumed to be the leaf
    best_mask = max(candidate_masks, key=lambda m: m['area'])
    x, y, bw, bh = best_mask['bbox']  # SAM returns [x, y, w, h]
    x, y, bw, bh = int(x), int(y), int(bw), int(bh)   # <-- the actual bug fix

    pad_x = int(bw * padding_frac)
    pad_y = int(bh * padding_frac)

    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y)

    return (int(x0), int(y0), int(x1), int(y1))


def process_dataset(input_dir, output_dir, mask_generator):
    class_names = [d for d in os.listdir(input_dir)
                   if os.path.isdir(os.path.join(input_dir, d))]

    total_processed = 0
    total_failed = 0

    for class_name in class_names:
        in_class_dir = os.path.join(input_dir, class_name)
        out_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(out_class_dir, exist_ok=True)

        image_files = [f for f in os.listdir(in_class_dir)
                       if Path(f).suffix.lower() in IMAGE_EXTENSIONS]

        # Skip files already processed (lets you resume after a crash/interrupt)
        already_done = set(os.listdir(out_class_dir)) if os.path.isdir(out_class_dir) else set()
        remaining = [f for f in image_files if f not in already_done]
        print(f"\nClass: {class_name} ({len(image_files)} total, {len(remaining)} remaining)")

        for fname in tqdm(remaining, desc=f"SAM cropping {class_name}"):
            img_path = os.path.join(in_class_dir, fname)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                total_failed += 1
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            try:
                bbox = get_leaf_bbox(img_rgb, mask_generator)
            except Exception as e:
                print(f"  SAM error on {fname}: {e}")
                bbox = None

            if bbox is None:
                cropped = img_bgr  # fallback: keep original image uncropped
            else:
                x0, y0, x1, y1 = bbox
                cropped = img_bgr[y0:y1, x0:x1]
                if cropped.size == 0:
                    cropped = img_bgr

            out_path = os.path.join(out_class_dir, fname)
            cv2.imwrite(out_path, cropped)
            total_processed += 1

    print(f"\nDone. Processed: {total_processed}, Failed/skipped: {total_failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='vit_b',
                         choices=['vit_b', 'vit_l', 'vit_h'])
    args = parser.parse_args()

    mask_generator = load_sam(args.checkpoint, args.model_type)
    process_dataset(args.input_dir, args.output_dir, mask_generator)


if __name__ == '__main__':
    main()