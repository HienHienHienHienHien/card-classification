import json
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data.preprocess import build_loaders, Metadata_dir, Num_classes
from src.models.complex_cnn import build_complex_cnn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 50
LEARNING_RATE = 3e-4
EARLY_STOPPING_PATIENCE = 8


def calculate_metrics(all_targets, all_preds):
    labels = list(range(Num_classes))

    return {
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision": precision_score(
            all_targets, all_preds,
            labels=labels,
            average="macro",
            zero_division=0
        ),
        "recall": recall_score(
            all_targets, all_preds,
            labels=labels,
            average="macro",
            zero_division=0
        ),
        "f1": f1_score(
            all_targets, all_preds,
            labels=labels,
            average="macro",
            zero_division=0
        ),
    }


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

    metrics = calculate_metrics(all_targets, all_preds)
    return total_loss / total, metrics


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
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

    metrics = calculate_metrics(all_targets, all_preds)
    return total_loss / total, metrics


def print_metrics(name, loss, metrics):
    print(
        f"{name} | Loss: {loss:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Precision: {metrics['precision']:.4f} | "
        f"Recall: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f}"
    )


def plot_history(history):
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train"]["loss"] for entry in history]
    val_loss = [entry["validation"]["loss"] for entry in history]
    train_acc = [entry["train"]["accuracy"] for entry in history]
    val_acc = [entry["validation"]["accuracy"] for entry in history]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="Train")
    plt.plot(epochs, val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "complex_cnn_curves.png", dpi=300)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "complex_cnn.pt"

    label_map_file = Metadata_dir / "label_map.json"
    with open(label_map_file, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    class_names = [
        name for name, _ in sorted(label_map.items(), key=lambda item: item[1])
    ]

    with open(OUTPUT_DIR / "complex_class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    loaders = build_loaders(
        model_type="scratch",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    model = build_complex_cnn(num_classes=Num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    best_val_acc = float("-inf")
    patience_counter = 0
    history = []
    best_epoch = 0

    try:
        for epoch in range(1, EPOCHS + 1):
            start_time = time.time()
            train_loss, train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_metrics = evaluate(
                model, val_loader, criterion, device
            )
            epoch_time = time.time() - start_time

            history.append({
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "duration_seconds": epoch_time,
                "train": {"loss": train_loss, **train_metrics},
                "validation": {"loss": val_loss, **val_metrics},
            })
            scheduler.step(val_loss)

            print(f"\nEpoch [{epoch:02d}/{EPOCHS:02d}]")
            print_metrics("Train", train_loss, train_metrics)
            print_metrics("Val  ", val_loss, val_metrics)
            print(
                f"Learning rate: {optimizer.param_groups[0]['lr']:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}.")
                break

        plot_history(history)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

        test_loss, test_metrics = evaluate(
            model, test_loader, criterion, device
        )

        results = {
            "model": "complex_cnn",
            "batch_size": BATCH_SIZE,
            "image_size": IMAGE_SIZE,
            "max_epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "parameters": sum(p.numel() for p in model.parameters()),
            "best_epoch": best_epoch,
            "best_validation_accuracy": best_val_acc,
            "test": {"loss": test_loss, **test_metrics},
            "history": history,
        }
        (OUTPUT_DIR / "complex_metrics.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )

        print("\n" + "-" * 50)
        print_metrics("Test", test_loss, test_metrics)
        print(f"Best model weights saved to: {checkpoint_path}")

    finally:
        for loader in loaders.values():
            loader.dataset.close()


if __name__ == "__main__":
    main()