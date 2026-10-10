from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import hashlib
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


Root = Path(__file__).resolve().parents[2]

Cleaned_archive = Root / "data/raw/dataset_cleaned.zip"
Metadata_dir = Root / "data/metadata"
Output_dir = Root / "data/preprocessing"

Image_size = 224
Batch_size = 32
Num_classes = 53

Split_names = ("train", "val", "test")
Columns = ["member", "class_name", "class_id"]

Imagenet_mean = (0.485, 0.456, 0.406)
Imagenet_std = (0.229, 0.224, 0.225)


def load_splits(metadata_dir=Metadata_dir):
    metadata_dir = Path(metadata_dir)

    return {split: pd.read_csv(metadata_dir / f"{split}.csv",
            usecols=list(Columns),
            dtype={ "member": "string",
                   "class_name": "string",
                   "class_id": "int64"},
            encoding="utf-8-sig")
            for split in Split_names}


def create_zip_index(zip_path, members):
    zip_path = Path(zip_path)

    if not zip_path.is_file():
        raise FileNotFoundError(f"Card ZIP not found: {zip_path}")

    requested_members = {str(member).strip().replace("\\", "/") for member in members}

    with ZipFile(zip_path) as archive:
        zip_members = set(archive.namelist())

    missing = requested_members - zip_members

    if missing:
        raise FileNotFoundError(f"ZIP is missing {len(missing)} images. Example: {sorted(missing)[:3]}")

    return {member: member for member in requested_members}


def create_transform(model_type, training, image_size=Image_size):
    if model_type not in ["scratch", "efficientnetb0"]:
        raise ValueError("model_type must be either scratch or efficientnetb0.")

    steps = [transforms.Resize((image_size, image_size))]

    if training:
        steps.extend([transforms.RandomRotation(degrees=10),
                      transforms.ColorJitter(brightness=0.1,contrast=0.1)])

    steps.append(transforms.ToTensor())

    if model_type == "efficientnetb0":
        steps.append(transforms.Normalize(Imagenet_mean, Imagenet_std))

    return transforms.Compose(steps)


class CardDataset(Dataset):
    def __init__(self, data, zip_path, index, transform):
        self.data = data.reset_index(drop=True)
        self.zip_path = Path(zip_path)
        self.index = index
        self.transform = transform
        self.archive = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, position):
        row = self.data.iloc[position]

        if self.archive is None:
            self.archive = ZipFile(self.zip_path)

        content = self.archive.read(self.index[row["member"]])

        with Image.open(BytesIO(content)) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {"image": image,
                "class_id": torch.tensor(int(row["class_id"]),dtype=torch.long),
                "member": row["member"]}

    def close(self):
        if self.archive is not None:
            self.archive.close()
            self.archive = None


def build_loaders(model_type="scratch",
                  metadata_dir=Metadata_dir,
                  zip_path=Cleaned_archive,
                  image_size=Image_size,
                  batch_size=Batch_size):
    
    if image_size < 32 or batch_size < 1:
        raise ValueError("image_size >= 32 and batch_size >= 1.")

    splits = load_splits(metadata_dir)

    members = pd.concat(splits.values())["member"]
    index = create_zip_index(zip_path, members)

    loaders = {}

    for name, data in splits.items():
        transform = create_transform(model_type=model_type,
                                     training=(name == "train"),
                                     image_size=image_size)

        dataset = CardDataset(data=data,
                              zip_path=zip_path,
                              index=index,transform=transform)

        loaders[name] = DataLoader(dataset,
                                   batch_size=batch_size,
                                   shuffle=(name == "train"),
                                   drop_last=False,num_workers=2)

    return loaders


def check_batches(loaders, model_type, image_size=Image_size):
    for name, loader in loaders.items():
        batch = next(iter(loader))
        size = len(batch["member"])

        assert batch["image"].shape == (size, 3, image_size, image_size)
        assert batch["class_id"].shape == (size,)

        assert batch["image"].dtype == torch.float32
        assert batch["class_id"].dtype == torch.int64

        assert torch.isfinite(batch["image"]).all()
        assert batch["class_id"].ge(0).all()
        assert batch["class_id"].lt(Num_classes).all()

        if model_type == "scratch":
            assert batch["image"].min() >= 0
            assert batch["image"].max() <= 1

        print(model_type, name, tuple(batch["image"].shape))

    for name in ["val", "test"]:
        dataset = loaders[name].dataset

        assert torch.equal(dataset[0]["image"],dataset[0]["image"])


def save_augmentation_preview(loader, model_type, output_path):
    dataset = loader.dataset
    figure, axes = plt.subplots(1, 4, figsize=(12, 3))

    for axis in axes:
        image = dataset[0]["image"].permute(1, 2, 0)
        if model_type == "efficientnetb0":
            image = (image * torch.tensor(Imagenet_std) + torch.tensor(Imagenet_mean))

        axis.imshow(image.clamp(0, 1).numpy())
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_config(Output_dir):
    label_map = json.loads((Metadata_dir / "label_map.json").read_text(encoding="utf-8"))

    assert len(label_map) == Num_classes, (f"label_map has {len(label_map)} classes, expected {Num_classes}.")

    config = {"image_size": [Image_size, Image_size],
              "color": "RGB",
              "batch_size": Batch_size,
              "target": "class_id",
              "num_classes": Num_classes,
              "label_map": label_map,
              "scratch": {"pixel_range": [0, 1]},
              "efficientnetb0": {"pixel_range_before_normalize": [0, 1],
                                 "mean": Imagenet_mean,
                                 "std": Imagenet_std},
              "train_augmentation": {"rotation_degrees": 10,
                                     "brightness": 0.1,
                                     "contrast": 0.1,
                                     "horizontal_flip": False},
              "validation_test_augmentation": False,
              "split_sha256": {name: hashlib.sha256((Metadata_dir / f"{name}.csv").read_bytes()).hexdigest()
                               for name in Split_names}}

    path = Output_dir / "preprocessing_config.json"
    path.write_text(json.dumps(config, indent=2), 
                    encoding="utf-8")


def main():
    Output_dir.mkdir(parents=True, exist_ok=True)
    save_config(Output_dir)

    for model_type in ["scratch", "efficientnetb0"]:
        loaders = build_loaders(model_type=model_type)

        try:
            check_batches(loaders, model_type)
            save_augmentation_preview(loaders["train"],
                                      model_type,
                                      Output_dir / f"augmentation_{model_type}.png")
        finally:
            for loader in loaders.values():
                loader.dataset.close()

    print("Preprocessing completed.")
    print("Output:", Output_dir)


if __name__ == "__main__":
    main()
