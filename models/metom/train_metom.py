"""
Fine-tune SakanaAI/Metom ViT for per-character Kuzushiji classification.

Data layout expected:
  data/gt/train/U+XXXX_*.png
  data/gt/val/U+XXXX_*.png

Class labels are extracted from the U+XXXX prefix of each filename.
The class-to-index mapping is saved inside the checkpoint so the
evaluate script can reconstruct it without any external file.

Usage:
  python models/metom/train_metom.py \
      --train_dir data/gt/train \
      --val_dir   data/gt/val \
      --output    models/metom/checkpoints/metom_finetuned.pth \
      --epochs 10 --batch_size 64 --lr 1e-4
"""

import argparse
import glob
import logging
import os

import torch
import torch.nn as nn
import wandb
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO = "SakanaAI/Metom"


def extract_label(filename: str) -> str:
    return os.path.basename(filename).split("_")[0]


def build_class_map(dirs):
    labels = set()
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.png")):
            labels.add(extract_label(p))
    classes = sorted(labels)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return classes, class_to_idx


def get_head(model):
    """Return (attr_name, module) for the classification head, regardless of attribute name."""
    for name in ("head", "classifier", "fc"):
        if hasattr(model, name):
            module = getattr(model, name)
            if isinstance(module, nn.Linear):
                return name, module
    # fallback: last Linear in named_modules
    last_name = last_module = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name, last_module = name, module
    if last_name is None:
        raise RuntimeError("Could not find a Linear classification head in MeTOM. Run print(model) to inspect.")
    return last_name, last_module


class KuzushijiDataset(Dataset):
    def __init__(self, image_dir, class_to_idx, processor):
        paths = glob.glob(os.path.join(image_dir, "*.png"))
        self.items = [(p, class_to_idx[extract_label(p)])
                      for p in paths if extract_label(p) in class_to_idx]
        self.processor = processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, label


def build_model(num_classes: int, device, dtype):
    processor = AutoImageProcessor.from_pretrained(REPO)
    model = AutoModel.from_pretrained(
        REPO,
        dtype=dtype,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)

    head_name, head_module = get_head(model)
    log.info(f"Classification head attribute: model.{head_name}  (in_features={head_module.in_features})")

    new_head = nn.Linear(head_module.in_features, num_classes).to(device=device, dtype=dtype)
    nn.init.trunc_normal_(new_head.weight, std=0.02)
    nn.init.zeros_(new_head.bias)

    # support dotted names like "classifier.out_proj"
    parts = head_name.split(".")
    obj = model
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_head)

    return processor, model, head_name


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    log.info(f"Device: {device}")

    log.info("Building class map …")
    classes, class_to_idx = build_class_map([args.train_dir, args.val_dir])
    num_classes = len(classes)
    log.info(f"Number of classes: {num_classes}")

    log.info("Loading MeTOM …")
    processor, model, head_name = build_model(num_classes, device, dtype)

    train_ds = KuzushijiDataset(args.train_dir, class_to_idx, processor)
    val_ds   = KuzushijiDataset(args.val_dir,   class_to_idx, processor)
    log.info(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # Differential LR: backbone uses lr*0.1, head uses lr
    head_param_ids = {id(p) for p in model.get_submodule(head_name).parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    head_params     = [p for p in model.parameters() if id(p) in head_param_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params,     "lr": args.lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # --- resume ---
    start_epoch = 1
    best_acc = 0.0
    resume_id = None
    last_ckpt = args.output.replace(".pth", "_last.pth")

    if args.resume and os.path.exists(last_ckpt):
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt["best_acc"]
        resume_id = ckpt.get("wandb_run_id")
        log.info(f"Resumed from epoch {ckpt['epoch']}  best_acc={best_acc:.4f}")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        id=resume_id,
        resume="allow" if args.resume else None,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_classes": num_classes,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "head_attr": head_name,
        },
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for pixels, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train"):
            pixels, labels = pixels.to(device, dtype=dtype), labels.to(device)
            optimizer.zero_grad()
            logits = model(pixels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * pixels.size(0)
        train_loss /= len(train_ds)

        # --- val ---
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for pixels, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} val"):
                pixels, labels = pixels.to(device, dtype=dtype), labels.to(device)
                logits = model(pixels)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        scheduler.step()

        log.info(f"Epoch {epoch}: loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc,
                   "lr_backbone": optimizer.param_groups[0]["lr"],
                   "lr_head":     optimizer.param_groups[1]["lr"]})

        # 毎エポック再開用チェックポイントを上書き保存
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "classes": classes,
            "class_to_idx": class_to_idx,
            "best_acc": best_acc,
            "val_acc": val_acc,
            "head_attr": head_name,
            "wandb_run_id": wandb.run.id,
        }, last_ckpt)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "class_to_idx": class_to_idx,
                "val_acc": val_acc,
                "head_attr": head_name,
            }, args.output)
            log.info(f"  → saved best checkpoint (val_acc={best_acc:.4f})")
            wandb.summary["best_val_acc"] = best_acc

    wandb.finish()
    log.info(f"Training done. Best val_acc={best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir",      default="data/gt/train")
    parser.add_argument("--val_dir",        default="data/gt/val")
    parser.add_argument("--output",         default="models/metom/checkpoints/metom_finetuned.pth")
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--wandb_project",  default="kuzushiji-metom")
    parser.add_argument("--wandb_run",      default=None)
    parser.add_argument("--resume",         action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
