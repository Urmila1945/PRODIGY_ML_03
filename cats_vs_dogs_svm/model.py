# model.py

import os
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# ----------------- Constants -----------------
IMG_SIZE = 64
CNN_MODEL_PATH = "cnn_model.h5"
SVM_MODEL_PATH = "svm_cat_dog_model.pkl"

# ----------------- SVM Model -----------------
def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(clf, SVM_MODEL_PATH)
    return clf, acc

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def load_svm_model():
    return joblib.load(SVM_MODEL_PATH)

# ----------------- CNN Model -----------------
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),  # RGB input
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_cnn(X, y, epochs=10, batch_size=32):
    # Ensure the input is normalized
    X = X.astype('float32') / 255.0
    y = np.array(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = create_cnn_model()

    # Save only the best model
    checkpoint = ModelCheckpoint(
        CNN_MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max'
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint]
    )
    return model, history

def load_cnn_model():
    return load_model(CNN_MODEL_PATH)
