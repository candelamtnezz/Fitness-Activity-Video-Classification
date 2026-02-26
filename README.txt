# Fitness Activity Video Classification (Task 3)

This project implements a video classification pipeline using **MoViNet (Mobile Video Networks)** for feature extraction and a custom Dense neural network for classifying fitness activities. It is designed to recognize exercises such as barbell curls, bench presses, deadlifts, and more.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Prerequisites](#prerequisites)
3. [Environment Setup](#environment-setup)
4. [Project Structure](#project-structure)
5. [Usage](#usage)

---

## 🚀 Project Overview
The notebook performs the following steps:
* **Data Loading:** Samples 16 frames from fitness videos and resizes them to 224x224.
* **Feature Extraction:** Utilizes a pre-trained MoViNet A0 model from TensorFlow Hub to convert video frames into high-level feature vectors.
* **Classification:** Trains a Sequential Keras model with a Dropout layer to prevent overfitting on the extracted features.
* **Evaluation:** Generates accuracy metrics and a confusion matrix to visualize model performance.

---

## 🛠 Prerequisites
* **Python:** 3.10
* **Conda:** (Anaconda or Miniconda)
* **Hardware:** A GPU is recommended for faster feature extraction, though the code is configured for CPU/Eager execution.

---

## ⚙️ Environment Setup

Follow these steps to create a dedicated Conda environment named `workout_vision`.

### 1. Create and Activate Environment
```bash
conda create -n workout_vision python=3.10 -y
conda activate workout_vision

pip install tensorflow==2.20.0 tensorflow-hub numpy==2.2.5
pip install opencv-python tqdm scikit-learn matplotlib seaborn
pip install ipykernel
python -m ipykernel install --user --name workout_vision --display-name "Python 3 (workout_vision)"