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

- Architecture: DenseNet169<br>
- Pretrained: ImageNet (fine-tuned)<br>
- Loss Function: CrossEntropyLoss<br>
- Optimizer: AdamW<br>
- Metric Used: F1 Score<br>
- Output Classes: 2<br>

---

### 1. Install Dependencies
pip install fastapi uvicorn torch torchvision pillow numpy python-multipart streamlit requests


📊 Future Improvements

Grad-CAM Visualization<br>
Use VLMs to get information about treatment and management,like a chatbot.<br>
Model Deployment (Render/AWS)<br>
Docker Containerization<br>
Threshold-based risk scoring<br>
Integration with Hospital Systems<br>
