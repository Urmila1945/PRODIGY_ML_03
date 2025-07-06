# split_data.py

import os
import shutil

SOURCE_DIR = "train"  # All images from Kaggle go here
DEST_DIR = "train"    # We'll create train/cat and train/dog under this

def create_folders():
    os.makedirs(os.path.join(DEST_DIR, "cat"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "dog"), exist_ok=True)

def move_images():
    for filename in os.listdir(SOURCE_DIR):
        if filename.startswith("cat"):
            shutil.move(os.path.join(SOURCE_DIR, filename),
                        os.path.join(DEST_DIR, "cat", filename))
        elif filename.startswith("dog"):
            shutil.move(os.path.join(SOURCE_DIR, filename),
                        os.path.join(DEST_DIR, "dog", filename))

if __name__ == "__main__":
    create_folders()
    move_images()
    print("✅ Images have been sorted into 'cat/' and 'dog/' folders.")
