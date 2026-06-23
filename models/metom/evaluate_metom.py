"""
Evaluate restoration quality using the fine-tuned MeTOM classifier.

Computes Top-1 accuracy for three image sets and compares them:
  - gt      : clean ground-truth images
  - lq      : damaged input images
  - restored: model-restored images (one or more directories)

Filename convention:
  gt/val/U+XXXX_<id>.png
  lq/val/U+XXXX_<id>_<DamageType>.png   (damage suffix stripped for matching)

Usage:
  python models/metom/evaluate_metom.py \
      --checkpoint models/metom/checkpoints/metom_finetuned.pth \
      --gt_dir     data/gt/val \
      --lq_dir     data/lq/val \
      --restored_dirs results/nafnet/val results/swinir/val \
      --restored_names NAFNet SwinIR \
      --output_csv results/ocr_comparison.csv
"""

import argparse
import csv
import glob
import logging
import os

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = "SakanaAI/Metom"

DAMAGE_SUFFIXES = [
    "_Ghosting", "_Missing", "_Abrasion", "_Stain", "_Transparent_Stain"
]


def strip_damage_suffix(stem: str) -> str:
    for suffix in DAMAGE_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def extract_label(filename: str) -> str:
    return os.path.basename(filename).split("_")[0]


class EvalDataset(Dataset):
    def __init__(self, image_dir, class_to_idx, processor):
        all_paths = glob.glob(os.path.join(image_dir, "*.png"))
        self.items = []
        for p in all_paths:
            label = extract_label(p)
            if label in class_to_idx:
                self.items.append((p, class_to_idx[label]))
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, label, path


def get_head(model):
    for name in ("head", "classifier", "fc"):
        if hasattr(model, name):
            module = getattr(model, name)
            if isinstance(module, nn.Linear):
                return name, module
    last_name = last_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name, last_module = name, module
    if last_name is None:
        raise RuntimeError("Could not find a Linear classification head in MeTOM.")
    return last_name, last_module


def load_model(checkpoint_path, device, dtype):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(classes)
    head_attr = ckpt.get("head_attr")

    processor = AutoImageProcessor.from_pretrained(REPO)
    model = AutoModel.from_pretrained(
        REPO,
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)

    # Use saved head_attr if available, otherwise auto-detect
    if head_attr is None:
        head_attr, head_module = get_head(model)
    else:
        head_module = model.get_submodule(head_attr)

    parts = head_attr.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    new_head = nn.Linear(head_module.in_features, num_classes).to(device=device, dtype=dtype)
    setattr(obj, parts[-1], new_head)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_acc = ckpt.get("val_acc", float("nan"))
    log.info(f"Loaded checkpoint: {checkpoint_path}  ({num_classes} classes, val_acc={val_acc:.4f})")
    return processor, model, classes, class_to_idx


def evaluate_dir(image_dir, class_to_idx, processor, model, device, dtype, batch_size, num_workers):
    ds = EvalDataset(image_dir, class_to_idx, processor)
    if len(ds) == 0:
        log.warning(f"No matching images in {image_dir}")
        return 0.0, []

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    correct = total = 0
    per_image = []

    with torch.no_grad():
        for pixels, labels, paths in tqdm(loader, desc=os.path.basename(image_dir)):
            pixels = pixels.to(device, dtype=dtype)
            labels = labels.to(device)
            logits = model(pixels)
            preds = logits.argmax(dim=1)
            for i in range(len(paths)):
                gt_label = extract_label(paths[i])
                is_correct = preds[i].item() == labels[i].item()
                correct += int(is_correct)
                total += 1
                per_image.append({
                    "path": paths[i],
                    "gt_label": gt_label,
                    "correct": is_correct,
                })

    acc = correct / total if total > 0 else 0.0
    return acc, per_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",     required=True, help="Path to fine-tuned checkpoint (.pth)")
    parser.add_argument("--gt_dir",         default="data/gt/val")
    parser.add_argument("--lq_dir",         default="data/lq/val")
    parser.add_argument("--restored_dirs",  nargs="*", default=[],
                        help="One or more restored image directories")
    parser.add_argument("--restored_names", nargs="*", default=[],
                        help="Display names for --restored_dirs (same order)")
    parser.add_argument("--output_csv",     default=None)
    parser.add_argument("--batch_size",     type=int, default=64)
    parser.add_argument("--num_workers",    type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    log.info(f"Device: {device}")

    processor, model, classes, class_to_idx = load_model(args.checkpoint, device, dtype)

    results = {}

    # Evaluate GT
    acc, rows = evaluate_dir(args.gt_dir, class_to_idx, processor, model,
                             device, dtype, args.batch_size, args.num_workers)
    results["GT"] = (acc, rows)
    log.info(f"GT       accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Evaluate LQ
    acc, rows = evaluate_dir(args.lq_dir, class_to_idx, processor, model,
                             device, dtype, args.batch_size, args.num_workers)
    results["LQ"] = (acc, rows)
    log.info(f"LQ       accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Evaluate restored dirs
    names = args.restored_names or [f"Restored_{i}" for i in range(len(args.restored_dirs))]
    for name, d in zip(names, args.restored_dirs):
        acc, rows = evaluate_dir(d, class_to_idx, processor, model,
                                 device, dtype, args.batch_size, args.num_workers)
        results[name] = (acc, rows)
        log.info(f"{name:<12} accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Summary table
    print("\n=== OCR Accuracy Comparison ===")
    print(f"{'Condition':<14} {'Accuracy':>10} {'vs LQ':>10}")
    lq_acc = results["LQ"][0]
    for key, (acc, _) in results.items():
        delta = f"{(acc - lq_acc)*100:+.2f}%" if key != "LQ" else "—"
        print(f"{key:<14} {acc*100:>9.2f}%  {delta:>10}")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["condition", "path", "gt_label", "correct"])
            for cond, (_, rows) in results.items():
                for r in rows:
                    writer.writerow([cond, r["path"], r["gt_label"], r["correct"]])
        log.info(f"Per-image results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
