import io
import json
import re
import zipfile
from hashlib import sha256
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

Root = Path(__file__).resolve().parents[2]
Raw_archive = Root / "data" / "raw" / "dataset.zip"
Cleaned_archive = Root / "data" / "raw" / "dataset_cleaned.zip"
Metadata_dir = Root / "data" / "metadata"

Image_size = (224, 224)
Expected_classes = 53
Random_state_fixed = 42
Near_duplicate_threshold = 8

Train_ratio = 0.70
Val_ratio = 0.15
Test_ratio = 0.15

Metadata_columns = ["member", "class_name", "class_id", "split"]
Metadata_dir.mkdir(parents=True, exist_ok=True)


def get_class_name(member):
    match = re.match(r"^(.+?)\s+\d+$", PurePosixPath(member).stem)
    if match is None:
        return None
    return " ".join(match.group(1).lower().split())


def get_dhash(image):
    image = image.convert("L")
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    image = image.resize((17, 16), resampling)
    pixels = np.asarray(image)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def inspect_dataset():
    if not Raw_archive.is_file():
        raise FileNotFoundError(f"Dataset archive was not found: {Raw_archive}")

    records = []
    skipped = []

    with zipfile.ZipFile(Raw_archive) as archive:
        for member in sorted(archive.namelist()):
            parts = PurePosixPath(member).parts

            if member.endswith("/") or any(p.startswith(".") or p == "__MACOSX" for p in parts):
                continue

            class_name = get_class_name(member)
            if class_name is None:
                skipped.append((member, "filename is not '<class name> <number>'"))
                continue

            try:
                image_bytes = archive.read(member)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.load()
                    width, height = image.size
                    channels = len(image.getbands())
                    image_dhash = get_dhash(image)
            except (OSError, ValueError, Image.DecompressionBombError):
                skipped.append((member, "not a readable image"))
                continue

            records.append({"member": member,
                            "class_name": class_name,
                            "sha256": sha256(image_bytes).hexdigest(),
                            "width": width,
                            "height": height,
                            "channels": channels,
                            "dhash": image_dhash,})

    metadata = pd.DataFrame(records)

    print("\nCARD DATASET INSPECTION")
    print(f"Valid images      : {len(metadata):,}")
    print(f"Skipped files     : {len(skipped):,}")
    for member, reason in skipped[:10]:
        print(f"  - {member}: {reason}")

    if not metadata.empty:
        print(f"Image dimensions  : {metadata[['width', 'height']].value_counts().to_dict()}")
        print(f"Channels          : {metadata['channels'].value_counts().sort_index().to_dict()}")
        print(f"Number of classes : {metadata['class_name'].nunique():,}")
        print("\nImages per class:")
        print(metadata["class_name"].value_counts().sort_index().to_string())

    return metadata


def remove_near_duplicates(metadata):
    hashes = np.stack(metadata["dhash"].to_numpy())
    parent = list(range(len(metadata)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first = find(first)
        second = find(second)

        if first != second:
            parent[second] = first

    for start in range(0, len(metadata), 32):
        current_hashes = hashes[start:start + 32]
        distances = np.count_nonzero(
            current_hashes[:, None, :] != hashes[None, :, :],
            axis=2)

        for row, column in np.argwhere(distances <= Near_duplicate_threshold):
            first = start + int(row)
            second = int(column)

            if second > first:
                union(first, second)

    groups = {}

    for index in range(len(metadata)):
        root = find(index)
        groups.setdefault(root, []).append(index)

    remove_indexes = []
    group_count = 0

    for group in groups.values():
        if len(group) < 2:
            continue

        group_count += 1
        group_classes = set(metadata.iloc[group]["class_name"])

        if len(group_classes) > 1:
            remove_indexes.extend(group)
        else:
            remove_indexes.extend(group[1:])

    return remove_indexes, group_count


def clean_metadata(metadata):
    before = len(metadata)

    labels_per_image = metadata.groupby("sha256")["class_name"].transform("nunique")
    conflicts = metadata[labels_per_image > 1]
    clean = metadata[labels_per_image == 1]

    clean = clean.drop_duplicates("sha256").copy()
    duplicates_removed = before - len(conflicts) - len(clean)

    near_duplicate_indexes, near_duplicate_groups = remove_near_duplicates(clean.reset_index(drop=True))
    clean = clean.reset_index(drop=True)
    clean = clean.drop(index=near_duplicate_indexes).copy()

    class_names = sorted(clean["class_name"].unique())
    class_to_id = {name: i for i, name in enumerate(class_names)}
    clean["class_id"] = clean["class_name"].map(class_to_id).astype("int64")

    with open(Metadata_dir / "label_map.json", "w", encoding="utf-8") as file:
        json.dump(class_to_id, file, ensure_ascii=False, indent=2)

    print("\nDATA CLEANING")
    print(f"Before cleaning   : {before:,}")
    print(f"After cleaning    : {len(clean):,}")
    print(f"Duplicate removed : {duplicates_removed:,}")
    print(f"Near duplicates   : {len(near_duplicate_indexes):,}")
    print(f"Near-dup groups   : {near_duplicate_groups:,}")
    print(f"Conflict files    : {len(conflicts):,}")
    print(f"Number of classes : {len(class_names):,} (expected {Expected_classes})")

    if len(class_names) != Expected_classes:
        print(f"WARNING: expected {Expected_classes} classes.")

    return clean


def create_cleaned_archive(metadata, clean_metadata_frame):
    keep_members = set(clean_metadata_frame["member"])
    image_members = set(metadata["member"])
    removed_images = 0

    Cleaned_archive.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(Raw_archive, "r") as source, zipfile.ZipFile(Cleaned_archive, "w", zipfile.ZIP_DEFLATED ) as target:
        for info in source.infolist():
            member = info.filename

            if member in image_members and member not in keep_members:
                removed_images += 1
                continue

            target.writestr(info, source.read(member))

    print("\nCLEANED ARCHIVE")
    print(f"Images removed   : {removed_images:,}")
    print(f"Cleaned archive  : {Cleaned_archive}")

    return Cleaned_archive


def create_splits(metadata):
    if metadata["class_id"].value_counts().min() < 7:
        raise ValueError("Every class needs at least 7 images to split 70/15/15.")

    train, temp = train_test_split(metadata,
                                   test_size=Val_ratio + Test_ratio,
                                   random_state=Random_state_fixed,
                                   stratify=metadata["class_id"])
    val, test = train_test_split(temp,
                                 test_size=Test_ratio / (Val_ratio + Test_ratio),
                                 random_state=Random_state_fixed,
                                 stratify=temp["class_id"])

    splits = {"train": train.assign(split="train"),
              "val": val.assign(split="val"),
              "test": test.assign(split="test")}

    print("\nDATA SPLIT")
    for name, data in splits.items():
        data = data[Metadata_columns].sort_values(["class_id", "member"])
        data.to_csv(Metadata_dir / f"{name}.csv", index=False)
        print(f"{name:<5}: {len(data):,} ({len(data) / len(metadata):.1%})")

    combined = pd.concat(splits.values())[Metadata_columns]
    combined = combined.sort_values(["class_id", "member"])
    combined.to_csv(Metadata_dir / "clean_metadata.csv", index=False)

    print("\nImages per class by split:")
    print(pd.crosstab(combined["class_name"], combined["split"])[["train", "val", "test"]].to_string())


def main():
    print("CARD CLASSIFICATION DATA PREPARATION")
    print(f"Dataset path: {Raw_archive}")
    print(f"Target size : {Image_size}")

    metadata = inspect_dataset()
    if metadata.empty:
        raise RuntimeError("No valid card images were found.")

    clean = clean_metadata(metadata)
    create_cleaned_archive(metadata, clean)
    create_splits(clean)
    print(f"\nSaved metadata to: {Metadata_dir}")
    print(f"Use this archive for training: {Cleaned_archive}")


if __name__ == "__main__":
    main()