# 🃏 53-Class Playing Card Classification

A deep learning project for classifying 53 playing card categories (52 standard cards and a joker), featuring a comparative performance evaluation between **Simple CNN**, **Complex CNN (ResNet-based)**, and **Transfer Learning (EfficientNet-B0)**.

🔗 **Repository:** [GitHub - card-classification](https://github.com/HienHienHienHienHien/card-classification.git)

### 1. Clone the Repository

```bash
git clone https://github.com/HienHienHienHienHien/card-classification.git
cd card-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```text
card-classification/
├── configs/                  # Configuration files and hyperparameters
├── data/
│   ├── raw/                  # Original and cleaned ZIP datasets
│   ├── metadata/             # CSV splits (train/val/test) & label mappings
│   └── preprocessing/        # Configs and augmentation previews
├── notebooks/
│   └── EDA.ipynb             # Exploratory Data Analysis & Dataset insights
├── outputs/                  # Training checkpoints (.pt) and loss/accuracy curves
├── src/
│   ├── data/                 # Dataset inspection, cleaning, and dataloaders
│   ├── models/               # Model architectures (Simple CNN, Complex CNN, Transfer)
│   └── training/             # Training scripts for each model type
├── requirements.txt          # Python package dependencies
└── README.md
```

---

## 📊 Dataset Preparation & Cleaning

The dataset consists of 53 classes. The preparation pipeline handles raw images directly inside a ZIP archive without requiring full manual disk extraction.

### 1. Pull Dataset via Git LFS (if tracked)

```bash
apt-get install git-lfs
git lfs install
git lfs pull
```

### 2. Run Exploratory Data Analysis (EDA)

Check `notebooks/EDA.ipynb` for dataset distributions and image sample previews.

### 3. Run Inspection and Cleaning

```bash
python src/data/inspect_data.py
```

### 4. Run Preprocessing and Loader Checks

```bash
python src/data/preprocess.py
```

### Data Preparation Pipeline

- **Deduplication:** Automatically removes exact byte-level duplicates using SHA-256 and conflicting label files before splitting.
- **Data Splitting:** Stratified split into **70% Train**, **15% Validation**, and **15% Test**, using a fixed random seed of `42`.
- **Outputs:** Generates `clean_metadata.csv`, `train.csv`, `val.csv`, `test.csv`, `label_map.json`, and `dataset_cleaned.zip`.

---

## 🚀 Training Models

The project compares three distinct model architectures. Run all commands from the project root.

### 1. Simple CNN (Baseline)

```bash
python -m src.training.train_simple_cnn
```

- **Architecture:** Lightweight 3-block convolutional neural network.
- **Purpose:** Serves as a fast baseline model for comparison.

### 2. Complex CNN (ResNet-style from Scratch)

```bash
python -m src.training.train_complex_cnn
```

- **Architecture:** Deep residual network trained completely from scratch.
- **Purpose:** Evaluates deep feature learning capabilities without pretrained weights.

### 3. Transfer Learning (EfficientNet-B0)

```bash
python -m src.training.train_transfer
```

- **Architecture:** Pretrained EfficientNet-B0 with ImageNet weights, using a two-phase training strategy: classifier-head training with the backbone frozen, followed by differential fine-tuning.
- **Purpose:** Leverages pretrained visual features to improve classification performance on the playing card dataset.

---

## 📈 Performance Summary

| Model Architecture | Training Strategy | Test Accuracy | F1-Score (Macro) |
| --- | --- | ---: | ---: |
| **Simple CNN** | From Scratch | ~71.6% | 0.72 |
| **Complex CNN** | From Scratch (ResNet Blocks) | ~86.2% | 0.86 |
| **Transfer Learning** | EfficientNet-B0 (Fine-tuned) | **~91.7%** | **0.92** |

---

## 📝 Notes

- Ensure that the dataset paths and configuration files match your local project structure before running the scripts.
- The reported metrics are included for model comparison and should be verified against the actual evaluation outputs.
- Training results may vary depending on the environment, random seed, and hyperparameter settings.
