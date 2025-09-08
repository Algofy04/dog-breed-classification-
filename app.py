import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ✅ Use full absolute paths to avoid FileNotFoundError
MODEL_PATH = "D:/project dog breed model/dog_breed_model_package/dog_breed_classifier_effnetb3.h5"
LABELS_PATH = "D:/project dog breed model/dog_breed_model_package/labels.txt"

# ✅ Check file existence
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: {MODEL_PATH}")
if not os.path.exists(LABELS_PATH):
    st.error(f"❌ Labels file not found: {LABELS_PATH}")

# ✅ Load model
model = tf.keras.models.load_model(MODEL_PATH)

# ✅ Load and clean class labels
with open(LABELS_PATH, "r") as f:
    raw_class_names = [line.strip() for line in f.readlines()]

class_names = []
for name in raw_class_names:
    if "-" in name:
        name = name.split("-", 1)[1]  # Remove prefix like n02085620-
    name = name.replace("_", " ").title()
    class_names.append(name)

# ✅ Streamlit UI
st.title("🐶 Dog Breed Classifier")
st.write("Upload a dog image and I'll predict its breed!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((300, 300))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict breed
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)
    predicted_label = class_names[predicted_index]

    st.success(f"🎯 Predicted Breed: *{predicted_label}*")
    st.info(f"📊 Confidence: *{confidence * 100:.2f}%*")