# 🫁 Lungs Cancer Classifier

Detect **Adenocarcinoma carcinoma cancer** from **CT scan images** using deep learning and MLOps principles.

---

## 🚀 Project Overview

The **Lungs Cancer Classifier** is an AI system that analyzes CT scan images to identify lung cancer.  
It uses **transfer learning** with the **VGG16** model and follows **MLOps** best practices to ensure reproducibility, scalability, and automation.

Deployed on [Hugging Face Spaces](#).

---

## ✨ Features

- 🖼️ CT Scan Image Classification (Adenocarcinoma carcinoma detection)
- 🧠 Transfer Learning with **VGG16**
- 🌐 Gradio web interface for easy interaction
- ⚙️ MLOps Pipeline:
  - DVC for data/model versioning
  - MLflow for experiment tracking
  - GitHub Actions for CICD
  - Custom logging
- 🚀 Deployment-ready structure

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- DVC
- MLflow
- Gradio
- YAML
- GitHub Actions
- Hugging Face Spaces

---

## 📁 Project Structure

```bash
├── artifacts/
├── config/
├── src/
│   ├── common/
│   ├── components/
│   ├── config/
│   ├── utils/
│   └── pipeline/
├── logs/
├── app.py
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── setup.py
├── README.md


