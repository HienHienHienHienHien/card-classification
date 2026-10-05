import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.preprocess import build_loaders, METADATA_DIR, NUM_CLASSES
from src.models.simple_cnn import build_simple_cnn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 20
LEARNING_RATE = 3e-4


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["class_id"].to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["class_id"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "simple_cnn.pt"

    # Lưu lại danh sách nhãn lớp từ label_map.json
    label_map_file = METADATA_DIR / "label_map.json"
    with open(label_map_file, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    with open(OUTPUT_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # 1. Khởi tạo DataLoader qua hàm tiền xử lý đã viết sẵn
    loaders = build_loaders(
        model_type="scratch",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # 2. Xây dựng mô hình
    model = build_simple_cnn(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    try:
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}"
            )

            # Lưu checkpoint có val_loss thấp nhất
            if val_acc > best_val_acc:
              best_val_acc = val_acc
              patience_counter = 0
              torch.save(model.state_dict(), checkpoint_path)
            else:
              patience_counter += 1

              if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        # 3. Đánh giá trên tập kiểm thử (Test Set)
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print("-" * 50)
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
        print(f"Best model weights saved to: {checkpoint_path}")

    finally:
        # Giải phóng tài nguyên zip
        for loader in loaders.values():
            loader.dataset.close()


if __name__ == "__main__":
    main()