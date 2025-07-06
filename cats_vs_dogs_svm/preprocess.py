# preprocess.py

import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import hog

CATEGORIES = ["cat", "dog"]
IMG_SIZE = 64  # Resize to 64x64

def extract_hog_features(image):
    return hog(image, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), visualize=False)

def load_data(datadir, limit_per_class=1000):
    features = []
    labels = []
    for category in CATEGORIES:
        path = os.path.join(datadir, category)
        class_num = CATEGORIES.index(category)
        for img_name in tqdm(os.listdir(path)[:limit_per_class], desc=f"Loading {category}s"):
            try:
                img_path = os.path.join(path, img_name)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                hog_features = extract_hog_features(resized_array)
                features.append(hog_features)
                labels.append(class_num)
            except Exception as e:
                continue
    return features, labels
