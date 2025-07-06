# utils.py

import matplotlib.pyplot as plt
import numpy as np

def show_samples(X, y, count=5):
    count = min(count, len(X))  # Prevent index out of bounds
    plt.figure(figsize=(10, 2))
    for i in range(count):
        plt.subplot(1, count, i+1)
        plt.imshow(X[i].reshape(64, 64), cmap='gray')
        plt.title("Cat" if y[i] == 0 else "Dog")
        plt.axis('off')
    plt.show()
