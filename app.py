
        # Import necessary libraries
import numpy as np  # For numerical computations
import pickle as pkl  # For loading saved feature vectors and filenames
import tensorflow as tf  # For deep learning models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input  # Pre-trained model and preprocessing
from tensorflow.keras.preprocessing import image  # Image processing utilities
from tensorflow.keras.layers import GlobalMaxPool2D  # Layer to extract maximum features from CNN output

from sklearn.neighbors import NearestNeighbors  # For similarity search
import os  # For file handling
from numpy.linalg import norm  # For normalizing feature vectors
import streamlit as st  # For building the web app

# Title of the Streamlit app
st.header('Fashion Recommendation System')

# Load precomputed image features and corresponding filenames
Image_features = pkl.load(open('Images_features.pkl', 'rb'))  # Feature vectors of all dataset images
filenames = pkl.load(open('filenames.pkl', 'rb'))  # List of image file paths

# Function to extract feature vector from an input image using the pre-trained ResNet50 model
def extract_features_from_images(image_path, model):
    # Load and resize image to match ResNet50 input size (224x224)
    img = image.load_img(image_path, target_size=(224, 224))
    
    # Convert image to a numpy array
    img_array = image.img_to_array(img)
    
    # Expand dimensions to match model input format (batch_size, height, width, channels)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image for ResNet50 (normalization, etc.)
    img_preprocess = preprocess_input(img_expand_dim)
    
    # Pass the image through the model and extract features
    result = model.predict(img_preprocess).flatten()
    
    # Normalize the feature vector for better similarity comparison
    norm_result = result / norm(result)
    
    return norm_result

# Load the pre-trained ResNet50 model (without the top classification layer)
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False  # Freeze model weights (no training needed)

# Add a GlobalMaxPool2D layer to reduce the feature map to a 1D vector
model = tf.keras.models.Sequential([
    model,
    GlobalMaxPool2D()
])

# Initialize the Nearest Neighbors model for finding similar images
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')

# Train the Nearest Neighbors model using precomputed image features
neighbors.fit(Image_features)

# File uploader in the Streamlit app
upload_file = st.file_uploader("Upload Image")

# If a file is uploaded
if upload_file is not None:
    # Save the uploaded file to the "upload" directory
    with open(os.path.join('upload', upload_file.name), 'wb') as f:
        f.write(upload_file.getbuffer())

    # Display the uploaded image
    st.subheader('Uploaded Image')
    st.image(upload_file)

    # Extract features from the uploaded image
    input_img_features = extract_features_from_images(upload_file, model)

    # Find the 5 most similar images using Nearest Neighbors
    distance, indices = neighbors.kneighbors([input_img_features])

    # Display recommended images
    st.subheader('Recommended Images')

    # Create columns for displaying images side by side
    col1, col2, col3, col4, col5 = st.columns(5)

    # Show recommended images in the columns
    with col1:
        st.image(filenames[indices[0][1]])
    with col2:
        st.image(filenames[indices[0][2]])
    with col3:
        st.image(filenames[indices[0][3]])
    with col4:
        st.image(filenames[indices[0][4]])
    with col5:
        st.image(filenames[indices[0][5]])
