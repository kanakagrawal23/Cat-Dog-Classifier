# 🐱🐶 Cat vs Dog Image Classification using Deep Learning

This project is a **binary image classification system** that classifies an input image as either a **Cat** or a **Dog** using **Convolutional Neural Networks (CNNs)** and **Transfer Learning**.  
The model is built and trained using **TensorFlow/Keras** on **Google Colab**.

---

## 📌 Project Overview

Image classification is a fundamental problem in computer vision. In this project, a pretrained deep learning model is fine-tuned to distinguish between images of cats and dogs with high accuracy.

To reduce training time and improve performance, **MobileNetV2**, a lightweight pretrained CNN, is used as the base model.

---

## 🧠 Key Concepts Used

- Convolutional Neural Networks (CNN)
- Transfer Learning
- Data Augmentation
- Binary Classification
- Image Preprocessing
- Model Evaluation & Prediction

---

## 📂 Dataset

- **Source:** Kaggle (Dogs vs Cats Dataset)
- **Structure:**
dataset/
├── train/
│ ├── cats/
│ └── dogs/
└── test/
├── cats/
└── dogs/


- Training Set:
- ~4000 images of cats
- ~4000 images of dogs
- Test Set:
- ~1000 images of cats
- ~1000 images of dogs

---

## 🏗️ Model Architecture

- **Base Model:** MobileNetV2 (pretrained on ImageNet)
- **Custom Layers Added:**
- Global Average Pooling
- Dense Layer (ReLU)
- Dropout (to reduce overfitting)
- Output Layer (Sigmoid activation)

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metric:** Accuracy  

---

## 🚀 How the Model Works

1. Images are resized to `224 × 224`
2. Pixel values are normalized
3. Data augmentation is applied to training images
4. Images are passed through the pretrained CNN
5. The final sigmoid output predicts:
 - `0 → Cat`
 - `1 → Dog`

---

## 🧪 Features

- Train a deep learning model on image data
- Evaluate model performance on unseen data
- Predict and classify a **single custom image**
- Display prediction with confidence
- Visualization of training & validation accuracy

---

## 📊 Results

- Achieved **~85–95% accuracy** within a few epochs
- Model generalizes well due to data augmentation and transfer learning

---

## 🖼️ Sample Prediction

The model can take any image as input and classify it as:

- **🐱 Cat**
- **🐶 Dog**

along with a confidence score.

---

## 🛠️ Tools & Technologies Used

- Python
- TensorFlow / Keras
- Google Colab
- Matplotlib
- NumPy

---

## 📁 Project Structure

Cat-vs-Dog-Classification/
├── Cat_vs_Dog_Classifier.ipynb
└── README.md


---

## 📝 Notes

- The `test` folder is used as validation data for simplicity.
- This project is intended for **learning and academic purposes**.

---

## 🙌 Acknowledgements

- Kaggle for the dataset
- TensorFlow documentation
- Google Colab for free GPU support

---

## ✨ Author

**Kanak Agrawal**  
Computer Science & Engineering (Data Science)

---

