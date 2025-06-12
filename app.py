import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="Wheat Disease Detector", page_icon="🌾")

# Load model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modelfinal.h5')

# Class labels (verify order matches training)
CLASS_NAMES = ["Crown and Root Rot", "Healthy Wheat", "Leaf Rust", "Wheat Loose Smut"]

# Image preprocessing (match training)
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Simple scaling (not VGG19-specific)
    return np.expand_dims(img_array, axis=0)

# Main app
def main():
    st.title("🌾 Wheat Disease Detection App")
    st.write("Upload an image of wheat plant to check for diseases")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess and predict
        processed_image = preprocess_image(image)
        model = load_model()
        predictions = model.predict(processed_image)
        
        # Get result
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        
        # Show only the predicted disease
        st.subheader("Analysis Results")
        st.metric(label="Predicted Condition", value=predicted_class)

# Run the app
if __name__ == "__main__":
    main()
