import json
from pathlib import Path


import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix


from src.data.preprocess import build_loaders, Metadata_dir, Num_classes
from src.models.transfer_cnn import build_transfer_cnn




PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


BATCH_SIZE = 32
IMAGE_SIZE = 224          
EPOCHS_HEAD = 5           # pha 1: đóng băng backbone, chỉ train head
EPOCHS_FINE = 50          # pha 2: fine-tune toàn mạng
EARLY_STOPPING_PATIENCE = 8              # early stopping ở pha 2
LABEL_SMOOTHING = 0.1


HEAD_LR = 1e-3
FINE_BACKBONE_LR = 1e-4
FINE_HEAD_LR = 5e-4
WEIGHT_DECAY_HEAD = 1e-4
WEIGHT_DECAY_FINE = 1e-2




def run_epoch(model, loader, criterion, device, optimizer=None,
              scaler=None, use_amp=False, freeze_bn=False):
    """Train nếu có optimizer, ngược lại evaluate. Trả về loss, acc."""
    training = optimizer is not None
    model.train() if training else model.eval()
    if training and freeze_bn:
        model.features.eval()  # giữ BatchNorm của backbone khi đang đóng băng


    total_loss = 0.0
    correct = 0
    total = 0


    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["class_id"].to(device)


            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)


            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()


            total_loss += loss.item() * targets.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)


    return total_loss / total, correct / total




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}")


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "transfer_cnn.pt"


    label_map_file = Metadata_dir / "label_map.json"
    with open(label_map_file, "r", encoding="utf-8") as file:
        label_map = json.load(file)


    class_names = [
        name for name, _ in sorted(label_map.items(), key=lambda item: item[1])
    ]
    with open(OUTPUT_DIR / "transfer_class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, ensure_ascii=False, indent=2)


    # model_type="resnet18" giữ nguyên như bản cũ (chuẩn hoá ImageNet).
    # Nếu preprocess.py có model_type riêng cho transfer learning thì đổi ở đây.
    loaders = build_loaders(
        model_type="resnet18",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]


    model = build_transfer_cnn(num_classes=Num_classes, pretrained=True).to(device)


    criterion_train = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    criterion_eval = nn.CrossEntropyLoss()  # loss thật để so sánh val/test
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


    best = {"acc": 0.0, "loss": float("inf")}


    def fit(epochs, optimizer, tag, freeze_bn, scheduler=None, patience=None):
        bad_epochs = 0
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion_train, device, optimizer,
                scaler=scaler, use_amp=use_amp, freeze_bn=freeze_bn,
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion_eval, device, use_amp=use_amp
            )


            if scheduler is not None:
                scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]


            print(
                f"[{tag}] Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )


            improved = val_acc > best["acc"] or (
                val_acc == best["acc"] and val_loss < best["loss"]
            )
            if improved:
                best["acc"], best["loss"] = val_acc, val_loss
                bad_epochs = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                bad_epochs += 1
                if patience and bad_epochs >= patience:
                    print(f"[{tag}] Early stopping at epoch {epoch}.")
                    break


    try:
        # Pha 1: đóng băng backbone, chỉ train head mới
        model.freeze_backbone()
        optimizer = AdamW(
            model.classifier.parameters(),
            lr=HEAD_LR,
            weight_decay=WEIGHT_DECAY_HEAD,
        )
        fit(EPOCHS_HEAD, optimizer, tag="head", freeze_bn=True)


        # Pha 2: mở toàn bộ mạng, LR nhỏ cho backbone + cosine scheduler
        model.unfreeze_backbone()
        optimizer = AdamW(
            [
                {"params": model.features.parameters(), "lr": FINE_BACKBONE_LR},
                {"params": model.classifier.parameters(), "lr": FINE_HEAD_LR},
            ],
            weight_decay=WEIGHT_DECAY_FINE,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FINE, eta_min=1e-6)
        fit(EPOCHS_FINE, optimizer, tag="fine", freeze_bn=False,
            scheduler=scheduler, patience=EARLY_STOPPING_PATIENCE)


        # Đánh giá trên tập test bằng model tốt nhất (theo val acc)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        test_loss, test_acc = run_epoch(
            model, test_loader, criterion_eval, device, use_amp=use_amp
        )


        print("-" * 50)
        print(f"Best Val Accuracy: {best['acc']:.4f}")
        print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
        print(f"Best model saved to: {checkpoint_path}")


        # Precision / Recall / F1
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                preds = model(batch["image"].to(device)).argmax(dim=1)
                y_pred += preds.cpu().tolist()
                y_true += batch["class_id"].tolist()


        labels = list(range(Num_classes))
        print(classification_report(
            y_true, y_pred, labels=labels,
            target_names=class_names, digits=4, zero_division=0,
        ))


        # Các cặp lớp bị nhầm nhiều nhất
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        for i in range(Num_classes):
            cm[i, i] = 0
        print("Các cặp bị nhầm nhiều nhất (thật -> dự đoán):")
        for idx in cm.flatten().argsort()[::-1][:10]:
            t, p = divmod(idx, Num_classes)
            if cm[t, p] > 0:
                print(f"  {class_names[t]} -> {class_names[p]}: {cm[t, p]}")


    finally:
        for loader in loaders.values():
            loader.dataset.close()




if __name__ == "__main__":
    main()
