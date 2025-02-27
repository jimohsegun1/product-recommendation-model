# import streamlit as st # type: ignore
# import os
# import pickle as pkl
# import numpy as np
# from sklearn.neighbors import NearestNeighbors
# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
# from tensorflow.keras.layers import GlobalMaxPool2D # type: ignore
# from tensorflow.keras.preprocessing import image # type: ignore
# from tensorflow.keras.models import Sequential # type: ignore
# from PIL import Image
# import imageio

# # Set page configuration
# st.set_page_config(page_title="Fashion Recommendation System", page_icon="🧥", layout="wide")

# # Custom CSS for background and footer
# st.markdown(
#     """
#     <style>
#     .st-emotion-cache-z5fcl4 {
#         width: 100%;
#         padding: 1rem 5rem;
#     }
#     body {
#         padding-top: 0 !important; 
#         background-color: #f7f7f7;
#     }
#     footer {
#         text-align: center;
#         font-size: small;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Header with style
# st.markdown(
#     """
#     <h1 style='text-align: center;'>
#         Fashion Recommendation System
#     </h1>
#     """,
#     unsafe_allow_html=True,
# )

# st.sidebar.title("Navigation")

# # Initialize session state for navigation
# if "nav" not in st.session_state:
#     st.session_state.nav = "Home"  # Default navigation

# # Custom CSS for buttons
# st.markdown(
#     """
#     <style>
#     div.stButton > button {
#         width: 200px;
#         height: 40px;
#         margin: 5px auto;
#         border-radius: 5px;
#         text-align: center;
#         font-weight: bold;
#         cursor: pointer;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Buttons for navigation with custom width and text alignment
# if st.sidebar.button("Home", key="home"):
#     st.session_state.nav = "Home"
# if st.sidebar.button("Recommendations", key="recommendations"):
#     st.session_state.nav = "Recommendations"
# if st.sidebar.button("Virtual Try-On", key="virtual_try_on"):
#     st.session_state.nav = "Virtual Try-On"
# if st.sidebar.button("About", key="about"):
#     st.session_state.nav = "About"

# # Access the current navigation state
# nav = st.session_state.nav



# # Load precomputed image features and filenames
# Image_features = pkl.load(open('Images_features.pkl', 'rb'))
# filenames = pkl.load(open('filenames.pkl', 'rb'))

# # Load ResNet50 model
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# base_model.trainable = False
# model = Sequential([
#     base_model,
#     GlobalMaxPool2D()
# ])

# # NearestNeighbors setup
# neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
# neighbors.fit(Image_features)

# # Function to handle image loading and AVIF conversion
# def load_image(image_path):
#     if image_path.endswith(".avif"):
#         avif_img = imageio.imread(image_path, format="avif")
#         img = Image.fromarray(avif_img).resize((224, 224))  # Resize to match model input
#     else:
#         img = image.load_img(image_path, target_size=(224, 224))
#     return img

# # Function to extract features from an image
# def extract_features_from_images(image_path, model):
#     img = load_image(image_path)
#     img_array = image.img_to_array(img)
#     img_expand_dim = np.expand_dims(img_array, axis=0)
#     img_preprocess = preprocess_input(img_expand_dim)
#     result = model.predict(img_preprocess).flatten()
#     norm_result = result / np.linalg.norm(result)
#     return norm_result

# # Home Page
# if nav == "Home":
#     st.subheader("Welcome to the Fashion Recommendation System!")
#     st.write(
#         """
#         Discover the perfect fashion matches for your style! Simply upload an image, and our intelligent system will recommend visually similar items to elevate your wardrobe.
#         """
#     )
    
#     st.markdown("### Why Fashion Recommendation?")
#     st.write(
#         """
#         - **Personalization**: Tailored recommendations that suit your unique style.
#         - **Convenience**: No need to browse endlessly; get relevant suggestions instantly.
#         - **Exploration**: Discover similar items and styles you may not have considered before.
#         """
#     )
    
#     st.markdown("### Use Cases")
#     st.write(
#         """
#         - **E-commerce**: Help customers find complementary or alternative fashion items.
#         - **Retail Stores**: Suggest similar items to upsell or cross-sell products.
#         - **Personal Styling**: Assist individuals in curating a cohesive wardrobe.
#         - **Trend Analysis**: Identify trending fashion items or styles.
#         """
#     )
    
#     with st.expander("How it Works"):
#         st.write(
#             """
#             - Upload an image of a fashion item.
#             - Our AI model analyzes the image and extracts its unique features.
#             - Using these features, the system identifies and recommends similar items from our database.
#             """
#         )

# # Recommendations Page
# elif nav == "Recommendations":
#     st.subheader("Upload an Image for Recommendations")
#     upload_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    
#     if upload_file is not None:
#         # Save uploaded file locally
#         file_path = os.path.join('upload', upload_file.name)
#         os.makedirs('upload', exist_ok=True)
#         with open(file_path, 'wb') as f:
#             f.write(upload_file.getbuffer())
        
#         st.subheader("Uploaded Image")
#         st.image(file_path, width=250)
        
#         with st.spinner("Analyzing the image..."):
#             input_img_features = extract_features_from_images(file_path, model)
#             distances, indices = neighbors.kneighbors([input_img_features])
#         st.success("Recommendations are ready!")
        
#         # Display recommendations in a grid
#         st.subheader("Recommended Items")
#         cols = st.columns(5)
#         for i, col in enumerate(cols):
#             if i < len(indices[0]) - 1:
#                 col.image(filenames[indices[0][i + 1]], caption=f"Recommendation {i+1}", width=180)

# # About Page
# elif nav == "About":
#     st.subheader("About this System")
#     st.write(
#         """
#         This Fashion Recommendation System leverages cutting-edge technologies to deliver accurate and efficient recommendations. Here's how it works:
#         """
#     )
    
#     st.markdown("### 1. TensorFlow and Keras for Feature Extraction")
#     st.write(
#         """
#         - **TensorFlow** is an open-source machine learning framework that powers our deep learning model.
#         - **Keras**, built on TensorFlow, provides a high-level interface to design, build, and train deep learning models efficiently.
#         - The system uses a **pre-trained ResNet50 model** (a deep convolutional neural network) to extract meaningful visual features from images.
#         - ResNet50 is particularly effective due to its ability to model intricate patterns in image data, making it ideal for tasks like identifying fashion styles and patterns.
#         """
#     )
    
#     st.markdown("### 2. Scikit-learn for Nearest Neighbor Search")
#     st.write(
#         """
#         - **Scikit-learn** is a powerful library for machine learning and data analysis.
#         - In this system, we use its **NearestNeighbors algorithm** to find the most visually similar images from the database.
#         - By analyzing the feature vectors extracted from ResNet50, the algorithm efficiently identifies the closest matches using metrics like Euclidean distance.
#         - This approach ensures quick and accurate recommendations, even with a large dataset of fashion items.
#         """
#     )
    
#     st.markdown("### 3. Streamlit for the Web Interface")
#     st.write(
#         """
#         - **Streamlit** is a modern Python library for building interactive and user-friendly web applications.
#         - It simplifies the deployment of machine learning models and makes it easy to create dynamic visualizations.
#         - The intuitive interface of this system, including file upload, navigation, and recommendations display, is powered by Streamlit.
#         - Streamlit's expanders, sliders, and columns allow for a clean, responsive design, enhancing user engagement and experience.
#         """
#     )
    
#     st.markdown("### Conclusion")
#     st.write(
#         """
#         Combining these technologies, this Fashion Recommendation System is a robust, scalable, and efficient solution for fashion-related applications.
#         Whether it's for e-commerce platforms, personal styling, or retail, this system delivers personalized and accurate results.
#         """
#     )


import streamlit as st
import os
import pickle as pkl
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.layers import GlobalMaxPool2D # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from PIL import Image
import imageio

# Set page configuration
st.set_page_config(page_title="Fashion Recommendation System", page_icon="🧥", layout="wide")

# Custom CSS for background and footer
st.markdown(
    """
    <style>
    body {
        background-color: #f7f7f7;
    }
    footer {
        text-align: center;
        margin-top: 20px;
        font-size: small;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header with style
st.markdown(
    """
    <h1 style='text-align: center;'>
        Fashion Recommendation System
    </h1>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")

# Initialize session state for navigation
if "nav" not in st.session_state:
    st.session_state.nav = "Home"  # Default navigation

# Custom CSS for buttons
st.markdown(
    """
    <style>
    div.stButton > button {
        width: 200px;
        height: 40px;
        margin: 5px auto;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Buttons for navigation with custom width and text alignment
if st.sidebar.button("Home", key="home"):
    st.session_state.nav = "Home"
if st.sidebar.button("Recommendations", key="recommendations"):
    st.session_state.nav = "Recommendations"
if st.sidebar.button("About", key="about"):
    st.session_state.nav = "About"

# Access the current navigation state
nav = st.session_state.nav

# Load precomputed image features and filenames
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = Sequential([
    base_model,
    GlobalMaxPool2D()
])

# NearestNeighbors setup
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# Function to handle image loading and AVIF conversion
def load_image(image_path):
    try:
        if image_path.endswith(".avif"):
            avif_img = imageio.imread(image_path, format="avif")
            img = Image.fromarray(avif_img).resize((224, 224))  # Resize to match model input
        else:
            img = image.load_img(image_path, target_size=(224, 224))
        return img
    except Exception as e:
        st.error(f"Error loading image: {image_path}. Details: {e}")
        return None

# Function to extract features from an image
def extract_features_from_images(image_path, model):
    img = load_image(image_path)
    if img is None:
        return None
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / np.linalg.norm(result)
    return norm_result

# Home Page
if nav == "Home":
    st.subheader("Welcome to the Fashion Recommendation System!")
    st.write(
        """
        Discover the perfect fashion matches for your style! Simply upload an image, and our intelligent system will recommend visually similar items to elevate your wardrobe.
        """
    )
    
    st.markdown("### Why Fashion Recommendation?")
    st.write(
        """
        - **Personalization**: Tailored recommendations that suit your unique style.
        - **Convenience**: No need to browse endlessly; get relevant suggestions instantly.
        - **Exploration**: Discover similar items and styles you may not have considered before.
        """
    )
    
    with st.expander("How it Works"):
        st.write(
            """
            - Upload an image of a fashion item.
            - Our AI model analyzes the image and extracts its unique features.
            - Using these features, the system identifies and recommends similar items from our database.
            """
        )

# Recommendations Page
elif nav == "Recommendations":
    st.subheader("Upload an Image for Recommendations")
    upload_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    
    if upload_file is not None:
        # Save uploaded file locally
        file_path = os.path.join('upload', upload_file.name)
        os.makedirs('upload', exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(upload_file.getbuffer())
        
        st.subheader("Uploaded Image")
        st.image(file_path, width=250)
        
        with st.spinner("Analyzing the image..."):
            input_img_features = extract_features_from_images(file_path, model)
            if input_img_features is None:
                st.error("Failed to process the uploaded image.")
            else:
                distances, indices = neighbors.kneighbors([input_img_features])
                st.success("Recommendations are ready!")
                
                # Display recommendations in a grid
                st.subheader("Recommended Items")
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    if i < len(indices[0]) - 1:
                        rec_image_path = filenames[indices[0][i + 1]]
                        if os.path.exists(rec_image_path):
                            col.image(rec_image_path, caption=f"Recommendation {i+1}", width=180)
                        else:
                            col.write(f"Image {i+1} not found.")

# About Page
elif nav == "About":
    st.subheader("About this System")
    st.write(
        """
        This Fashion Recommendation System leverages cutting-edge technologies to deliver accurate and efficient recommendations.
        """
    )
