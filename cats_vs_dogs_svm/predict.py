# predict.py

import cv2
import numpy as np
from skimage.feature import hog

from model import train_svm, load_svm_model, load_cnn_model
from preprocess import load_data

IMG_SIZE = 64
LABELS = ["Cat 🐱", "Dog 🐶"]

# ----------- Preprocessing -----------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img

# ----------- Predict Using SVM -----------
def predict_svm(image_path):
    model = load_svm_model()
    img = preprocess_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    ).reshape(1, -1)

    prediction = model.predict(hog_features)[0]
    prob = model.predict_proba(hog_features)[0][prediction]
    label = LABELS[prediction]

    return label, round(prob * 100, 2)

# ----------- Predict Using CNN -----------
def predict_cnn(image_path):
    model = load_cnn_model()
    img = preprocess_image(image_path)
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)

    prob = model.predict(img)[0][0]
    prediction = 1 if prob > 0.5 else 0
    confidence = prob if prediction == 1 else 1 - prob
    label = LABELS[prediction]

    return label, round(confidence * 100, 2)

# ----------- Generalized Predict Function -----------
def predict(image_path, model_type="svm"):
    if model_type.lower() == "svm":
        return predict_svm(image_path)
    elif model_type.lower() == "cnn":
        return predict_cnn(image_path)
    else:
        raise ValueError("Invalid model_type. Choose 'svm' or 'cnn'")

# ----------- Legacy Direct Predict (Optional Test) -----------
def predict_image(model, image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).flatten().reshape(1, -1)
    prediction = model.predict(img_resized)[0]
    return LABELS[prediction]
