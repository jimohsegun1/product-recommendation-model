import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm

# Define paths
IMAGE_DIR = os.path.abspath("images")
FEATURES_PATH = os.path.abspath("Images_features.pkl")
FILENAMES_PATH = os.path.abspath("filenames.pkl")

# Extract Filenames
filenames = [os.path.join(IMAGE_DIR, file) for file in os.listdir(IMAGE_DIR) if file.endswith((".jpg", ".png"))]

# Load Pretrained Model
model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Function to extract image features
def extract_features(image_path, model):
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return None

    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    return result / norm(result)

# Extract features for all images
image_features = [extract_features(file, model) for file in filenames]
image_features = [feat for feat in image_features if feat is not None]  # Remove None values

# Save features and filenames
with open(FEATURES_PATH, "wb") as f:
    pkl.dump(image_features, f)

with open(FILENAMES_PATH, "wb") as f:
    pkl.dump(filenames, f)

print("Feature extraction and saving complete.")
