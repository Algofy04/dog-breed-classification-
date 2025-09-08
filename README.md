Dog Breed Classification Project

🐶 Project Overview

This project is a deep learning-based dog breed classification system built using EfficientNetB3 with TensorFlow/Keras. The model can accurately identify 120 dog breeds from images. The system can be used for educational purposes, pet recognition applications, or AI-powered mobile/web apps.


---

🔧 Tools & Technologies

Programming Language: Python 3.12

Deep Learning Frameworks: TensorFlow, Keras

Model Architecture: EfficientNetB3 (Convolutional Neural Network)

Data Handling: NumPy, Pandas

Image Processing: OpenCV, PIL

Visualization: Matplotlib, Seaborn

Deployment: Streamlit (optional)

Version Control: Git, GitHub



---

📂 Dataset

Number of Classes: 120 dog breeds

Training Images: 14,355

Validation Images: 3,200

Test Images: 3,025

Source: Publicly available dog breed datasets (e.g., Kaggle)

Data Preprocessing:

Images resized to fit EfficientNetB3 input (300x300 or 380x380)

Normalization of pixel values

Data augmentation: rotation, flipping, zoom, etc.




---

🧠 Model Architecture

Base model: EfficientNetB3 (pre-trained on ImageNet)

Added Global Average Pooling Layer

Added Dense Layers with softmax activation for 120-class classification

Optimizer: Adam

Loss function: Categorical Crossentropy

Metrics: Accuracy



---

⚡ Features

Predicts dog breed from a single image

Provides top-N predictions with probabilities

Easy to deploy using Streamlit

High accuracy (90%+ achieved on validation data)
