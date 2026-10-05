# Pneumonia & Tuberculosis Classification using Deep Learning

## 📌 Project Overview
This project builds a machine learning model that classifies chest X-ray images into three categories:
- **Normal**
- **Pneumonia**
- **Tuberculosis**

The aim is to assist doctors in making faster and more accurate diagnoses by leveraging deep learning for medical image classification.

---

## 🎯 Problem Statement
> Build a machine learning model that can classify images into three categories: Normal, Pneumonia, and Tuberculosis—helping doctors make faster and more accurate diagnoses.

We focus not only on building a performant model, but also on **data preprocessing, evaluation metrics, and ethical considerations** in AI for healthcare.

---

## 🗂️ Dataset
- Images are collected from publicly available medical datasets (e.g., Kaggle or GitHub repositories).
- Data is split into:
  - **70% Training**
  - **15% Validation**
  - **15% Testing**

---

## ⚙️ Methodology

### 🔹 Data Preprocessing
- Images resized and normalized.
- Applied augmentations (flips, rotations, shifts) for generalization.
- Ensured class balance across splits.

### 🔹 Model
- Neural network (custom CNN baseline).
- Dropout, Batch Normalization used to prevent overfitting.
- Can be extended to advanced models (ResNet, EfficientNet).

### 🔹 Training
- Optimizer: Adam
- Loss: CrossEntropyLoss
- Scheduler: StepLR for learning rate decay.

### 🔹 Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC-AUC

---

## 📊 Results
- Training and validation loss curves.
- Confusion matrices for error analysis.
- Metrics reported across all classes.

---

## 💡 Key Insights
- Pneumonia detection works reasonably well.
- Tuberculosis classification shows inconsistency (due to smaller dataset size and overlapping features with Pneumonia).
- Validation accuracy is significantly lower than training → possible **overfitting**.
- Data imbalance (Normal vs Pneumonia vs TB) needs addressing.

---

## 🤔 Think Beyond Code

### ⚠️ Bias in Data
- Datasets may be biased toward specific hospitals, regions, or age groups → model might underperform on underrepresented demographics.
- Imbalanced class distribution (more Pneumonia images than TB) can skew predictions.

### 🔍 Interpretability
- Saliency maps / Grad-CAM should be added to visualize *why* the model makes a decision.
- Doctors need to trust not only the output but also the reasoning.

### 🏥 Ethical Considerations
- Model is **assistive**, not a replacement for radiologists.
- False negatives (missing Pneumonia/TB) are dangerous → must prioritize recall over raw accuracy.
- Deployment must consider patient privacy, HIPAA compliance, and bias mitigation.

---

## 🚀 How to Run

```bash
# Clone repo
git clone https://github.com/yourusername/Pneumonia_TB_classification.git
cd Pneumonia_TB_classification

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook Pneumonia_TB_classification_NN_fixed.ipynb
