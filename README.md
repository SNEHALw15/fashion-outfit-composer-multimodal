# Multimodal Fashion Outfit Compatibility Prediction

## 📌 Project Title
Multimodal Fashion Outfit Compatibility Prediction using Deep Learning

---

## ❓ Problem Statement
The goal of this project is to predict whether a given fashion outfit is compatible or not.  
Each outfit consists of multiple fashion items, and the task is to determine if these items go well together.

This is formulated as a **binary classification problem**, where:
- **1 → Compatible outfit**
- **0 → Incompatible outfit**

---

## 📂 Dataset
We use the **Polyvore Outfit Dataset (Disjoint Version)**.

### Key Features:
- Contains fashion outfits with multiple items
- Includes:
  - Item images
  - Metadata (category, title)
- Split into:
  - `train.json`
  - `valid.json`
  - `test.json`
- Disjoint splits ensure no overlap between outfits

---

## 🧠 Model Architecture

This is a **multimodal deep learning model** combining image and text features.

### 🔹 Image Feature Extraction
- Model: **EfficientNet-B2 (pretrained)**
- Used as a **feature extractor (frozen)**
- Output feature size: **1408**

### 🔹 Text Feature Extraction
- Model: **BERT (bert-base-uncased)**
- Input: Combined text (category + title)
- Output: **768-dimensional embedding**
- **Fine-tuned during training**

### 🔹 Feature Fusion
- Image + Text features are **concatenated**
- Passed through a **Multi-Layer Perceptron (MLP)**

### 🔹 Classification Head
- Linear Layer → ReLU → Linear Layer → Sigmoid
- Output: **Compatibility score (0–1)**

---

## 📊 Results

| Model Type        | Accuracy |
|------------------|---------|
| Image-only model | 71.35%  |
| Multimodal model | ~79%    |

Additional Metric:
- **AUC Score: 0.875**

---

## ⚙️ How to Run

### 1. Clone Repository

2. Install Dependencies
   pip install torch torchvision transformers pillow scikit-learn

3. Dataset Setup
Place dataset files:
train_with_negatives.json
polyvore_item_metadata.json
images/ folder

4. Train Model

5. Evaluate Model
6. 🚀 Key Contributions
Multimodal fusion of image + text
Use of EfficientNet-B2 + BERT
Improved performance over image-only baseline
Achieved strong classification performance on Polyvore dataset
      
