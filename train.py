import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import InstagramPostDataset, simple_collate
from models import MultiModalRegressor, get_device


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        for sample in batch:
            images = [img.to(device) for img in sample["images"]]
            text = sample["text"].to(device)
            meta = sample["meta"].to(device)
            target = sample["target"].to(device)

            optimizer.zero_grad()
            pred = model(images, text, meta)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            for sample in batch:
                images = [img.to(device) for img in sample["images"]]
                text = sample["text"].to(device)
                meta = sample["meta"].to(device)
                target = sample["target"].to(device)
                pred = model(images, text, meta)
                loss = criterion(pred, target)
                total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="Train multi-modal regressor")
    parser.add_argument("--processed", default="processed/processed_data.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--model_out", default="processed/model.pt")
    args = parser.parse_args()

    set_seed()
    device = get_device()
    print(f"Using device: {device}")

    dataset = InstagramPostDataset(args.processed, device=device)
    text_dim = len(dataset[0]["text"])
    meta_dim = len(dataset[0]["meta"])

    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=simple_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=simple_collate)

    model = MultiModalRegressor(text_dim=text_dim, meta_dim=meta_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val = float("inf")
    artifacts_path = Path(args.model_out).with_name("train_artifacts.pt")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "text_dim": text_dim,
                "meta_dim": meta_dim,
            }, args.model_out)
            print(f"Saved best model to {args.model_out}")

            torch.save({
                "model_state": model.state_dict(),
                "text_dim": text_dim,
                "meta_dim": meta_dim,
                "device": str(device),
                "val_indices": getattr(val_ds, "indices", None),
                "train_indices": getattr(train_ds, "indices", None),
                "batch_size": args.batch_size,
                "collate_fn": "simple_collate",
                "processed_path": args.processed,
            }, artifacts_path)
            print(f"Saved training artifacts (model, val split, device) to {artifacts_path}")
    return model, val_loader, device

if __name__ == "__main__":
    main()
