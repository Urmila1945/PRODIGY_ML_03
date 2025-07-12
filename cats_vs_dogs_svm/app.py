import streamlit as st
import numpy as np
import cv2
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from model import train_svm, load_svm_model, train_cnn, load_cnn_model
from predict import predict
from preprocess import load_data

# -------------------- Configuration --------------------
CATEGORIES = ["cat", "dog"]
IMG_SIZE = 64
DATASET_PATH = "train"

st.set_page_config(page_title="Cat vs Dog Classifier", layout="wide")
st.title("🐾 Cat vs Dog Classifier")
st.markdown("Classify cat vs dog images using SVM or CNN")

# -------------------- Sidebar --------------------
model_type = st.sidebar.radio("Choose Model", ["SVM (HOG)", "CNN (Keras)"])
selected_model_type = "svm" if "SVM" in model_type else "cnn"

if st.sidebar.button("🔁 Train / Retrain Model"):
    st.info("📥 Loading and preprocessing data...")

    features, labels = load_data(DATASET_PATH, limit_per_class=500)

    if not features or len(features) < 2:
        st.error("❌ Not enough images to train. Please check your folders.")
    else:
        X = np.array(features)
        y = np.array(labels)

        if selected_model_type == "svm":
            st.info("🧠 Training SVM Model...")
            model = train_svm(X, y)[0]
            joblib.dump(model, "svm_cat_dog_model.pkl")
            st.success("✅ SVM model trained and saved!")

        else:
            st.info("🧠 Training CNN Model...")

            # Load RGB images for CNN (3 channels)
            cnn_X = []
            cnn_y = []
            for category in CATEGORIES:
                path = os.path.join(DATASET_PATH, category)
                class_num = CATEGORIES.index(category)
                for img_name in os.listdir(path)[:500]:
                    try:
                        img_path = os.path.join(path, img_name)
                        img_array = cv2.imread(img_path)  # Read as RGB
                        resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        cnn_X.append(resized_array)
                        cnn_y.append(class_num)
                    except Exception:
                        continue

            cnn_X = np.array(cnn_X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # RGB
            cnn_X = cnn_X / 255.0  # Normalize
            cnn_y = np.array(cnn_y)

            model, history = train_cnn(cnn_X, cnn_y, epochs=5)
            model.save("cnn_model.h5")
            st.success("✅ CNN model trained and saved!")

# -------------------- Upload and Predict --------------------
st.header("📤 Upload Image for Prediction")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🎯 Predict"):
        if selected_model_type == "svm" and not os.path.exists("svm_cat_dog_model.pkl"):
            st.warning("Train the SVM model first.")
        elif selected_model_type == "cnn" and not os.path.exists("cnn_model.h5"):
            st.warning("Train the CNN model first.")
        else:
            label, confidence = predict(temp_path, model_type=selected_model_type)
            st.success(f"Prediction: **{label}** with {confidence}% confidence")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, OpenCV, and Scikit-Learn/Keras")
