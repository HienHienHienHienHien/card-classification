import json
from pathlib import Path

import tensorflow as tf

from src.models.simple_cnn import build_simple_cnn


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs" / "checkpoints"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30


def load_dataset(folder, class_names=None, shuffle=False):
    return tf.keras.utils.image_dataset_from_directory(
        folder,
        class_names=class_names,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=shuffle
    )


def main():
    train_ds = load_dataset(DATA_DIR / "train", shuffle=True)
    class_names = train_ds.class_names

    if len(class_names) != 53:
        raise ValueError(f"Cần 53 lớp, nhưng tìm thấy {len(class_names)} lớp.")

    valid_ds = load_dataset(
        DATA_DIR / "valid",
        class_names=class_names
    )
    test_ds = load_dataset(
        DATA_DIR / "test",
        class_names=class_names
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "simple_cnn.keras"

    with open(OUTPUT_DIR / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, ensure_ascii=False, indent=2)

    model = build_simple_cnn(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        num_classes=len(class_names)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    best_model = tf.keras.models.load_model(model_path)
    test_loss, test_accuracy = best_model.evaluate(test_ds)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()