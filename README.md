# 🩺 Oral Cancer Detection API using DenseNet169

A Deep Learning-based Oral Cancer Classification system built using **PyTorch (DenseNet169)** and deployed using **FastAPI** with a **Streamlit frontend**.

This system classifies oral cavity images into:

- ✅ Cancerous
- ✅ Non-Cancerous

---

## 🚀 Project Overview

This project uses a fine-tuned DenseNet169 Convolutional Neural Network trained on oral cavity images.

- Dataset split: 70% Train, 15% Validation, 15% Test
- 5-Fold Cross Validation used
- Best fold selected based on F1-score
- Model saved as `best_cv_model.pkl`

The API allows users to upload an image and receive:

- Predicted Class
- Confidence Score

---

## 🧠 Model Details

- Architecture: DenseNet169
- Pretrained: ImageNet (fine-tuned)
- Loss Function: CrossEntropyLoss
- Optimizer: AdamW
- Metric Used: F1 Score
- Output Classes: 2

---

## 📁 Project Structure
