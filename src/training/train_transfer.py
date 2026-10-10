import time
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.preprocess import build_loaders, Num_classes
from src.models.transfer import build_transfer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS_HEAD = 5
EPOCHS_FINE = 40
HEAD_LR = 1e-3
FINE_BACKBONE_LR = 1e-5
FINE_HEAD_LR = 1e-4
EARLY_STOPPING_PATIENCE = 5


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["class_id"].to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * targets.size(0)
            total += targets.size(0)
            all_targets.extend(targets.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    return total_loss / total, accuracy, all_targets, all_preds


def train_epoch(model, loader, criterion, optimizer, device, freeze_bn=False):
    model.train()
    if freeze_bn:
        model.features.eval()

    total_loss = 0.0
    total = 0
    all_targets = []
    all_preds = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["class_id"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)

        total_loss += loss.item() * targets.size(0)
        total += targets.size(0)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    accuracy = accuracy_score(all_targets, all_preds)
    return total_loss / total, accuracy


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transfer_cnn_curves.png", dpi=300)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "transfer_cnn.pt"

    loaders = build_loaders(
        model_type="efficientnetb0",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    model = build_transfer(num_classes=Num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()

    best_val_acc = float("-inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    try:
        # Phase 1: freeze backbone, train head only
        # Phase 2: unfreeze all, fine-tune with CosineAnnealingLR + early stopping
        for phase in ("head", "fine"):
            if phase == "head":
                model.freeze_backbone()
                optimizer = AdamW(model.classifier.parameters(), lr=HEAD_LR, weight_decay=1e-4)
                scheduler = None
                num_epochs = EPOCHS_HEAD
                freeze_bn = True
            else:
                model.unfreeze_backbone()
                optimizer = AdamW(
                    [
                        {"params": model.features.parameters(), "lr": FINE_BACKBONE_LR},
                        {"params": model.classifier.parameters(), "lr": FINE_HEAD_LR},
                    ],
                    weight_decay=1e-4
                )
                scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FINE, eta_min=1e-6)
                num_epochs = EPOCHS_FINE
                freeze_bn = False
                patience_counter = 0

            for epoch in range(1, num_epochs + 1):
                start_time = time.time()

                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, device, freeze_bn
                )

                val_loss, val_acc, _, _ = evaluate(
                    model, val_loader, criterion, device
                )

                if scheduler is not None:
                    scheduler.step()

                train_time = time.time() - start_time

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["train_acc"].append(train_acc)
                history["val_acc"].append(val_acc)

                print(
                    f"[{phase}] Epoch [{epoch:02d}/{num_epochs:02d}] | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                    f"Time: {train_time:.1f}s"
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    torch.save(model.state_dict(), checkpoint_path)
                else:
                    patience_counter += 1

                if phase == "fine" and patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        plot_history(history)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        test_loss, test_acc, all_targets, all_preds = evaluate(
            model, test_loader, criterion, device
        )

        labels = list(range(Num_classes))

        precision = precision_score(all_targets, all_preds, labels=labels, average="macro", zero_division=0)
        recall = recall_score(all_targets, all_preds, labels=labels, average="macro", zero_division=0)
        f1 = f1_score(all_targets, all_preds, labels=labels, average="macro", zero_division=0)

        print("\n" + "-" * 50)
        print(f"Best Val Accuracy: {best_val_acc:.4f}")
        print("Final Test Results")
        print(f"Loss      : {test_loss:.4f}")
        print(f"Accuracy  : {test_acc:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print(f"Best Val Accuracy: {best_val_acc:.4f}")

    finally:
        for loader in loaders.values():
            loader.dataset.close()


if __name__ == "__main__":
    main()