"""
🌿 PLANTDOCTOR - Streamlit Test App
Test your plant disease detection model
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="🌿 PlantDoctor",
    page_icon="🌿",
    layout="centered"
)

# Title
st.title("🌿 PlantDoctor - Plant Disease Detection")
st.markdown("Upload a leaf image to diagnose plant diseases")


# Load model and class names
@st.cache_resource
def load_model():
    """Load the TFLite model and class names"""
    try:
        # Path to your model files
        model_path = "plantdoctor_model.tflite"
        labels_path = "class_names.txt"

        # Check if files exist
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.info("Please place your model file in the same directory as this script")
            return None, None, None, None

        if not os.path.exists(labels_path):
            st.error(f"❌ Labels file not found: {labels_path}")
            return None, None, None, None

        # Load TFLite model using TensorFlow Lite
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Load class names
        with open(labels_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

        st.success(f"✅ Model loaded successfully! Found {len(class_names)} classes")

        return interpreter, class_names, input_details, output_details

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None


# Preprocess image
def preprocess_image(image, input_details):
    """Preprocess image for model input"""
    # Get input shape
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]

    # Resize image
    image = image.resize((width, height))

    # Convert to array and normalize
    image_array = np.array(image, dtype=np.float32)

    # Normalize to [-1, 1] (matching MobileNetV2 preprocessing)
    image_array = (image_array / 127.5) - 1

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# Predict
def predict(interpreter, image_array, input_details, output_details, class_names):
    """Run inference on the image"""
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run inference
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get prediction
    prediction = output_data[0]
    class_id = np.argmax(prediction)
    confidence = prediction[class_id]

    return class_names[class_id], confidence, prediction


# Main app
def main():
    # Load model
    interpreter, class_names, input_details, output_details = load_model()

    if interpreter is None:
        st.warning("⚠️ Please place the model files in the correct location")
        st.info("""
        Required files:
        - `plantdoctor_model.tflite` (your trained model)
        - `class_names.txt` (list of class names)

        Make sure both files are in the same directory as this script.
        """)
        return

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📸 Camera", "📁 Upload Image", "ℹ️ About"])

    # Tab 1: Camera
    with tab1:
        st.subheader("📸 Take a Photo")
        st.caption("Use your camera to take a photo of a plant leaf")

        # Camera input
        camera_image = st.camera_input("Take a picture")

        if camera_image is not None:
            # Display uploaded image
            st.image(camera_image, caption="Captured Image", use_container_width=True)

            # Process and predict
            with st.spinner("🔄 Analyzing..."):
                # Open image
                image = Image.open(camera_image)

                # Convert to RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Preprocess
                image_array = preprocess_image(image, input_details)

                # Predict
                class_name, confidence, prediction = predict(
                    interpreter, image_array, input_details, output_details, class_names
                )

                # Determine if healthy or diseased
                is_healthy = 'healthy' in class_name.lower()

                # Show result
                if is_healthy:
                    st.success(f"✅ **Diagnosis:** {class_name}")
                else:
                    st.warning(f"⚠️ **Diagnosis:** {class_name}")

                # Show confidence
                st.metric("Confidence", f"{confidence * 100:.1f}%")

                # Show all probabilities
                with st.expander("📊 View all probabilities"):
                    sorted_indices = np.argsort(prediction)[::-1]
                    for idx in sorted_indices[:5]:
                        prob = prediction[idx] * 100
                        bar = "█" * int(prob / 5)
                        st.write(f"{class_names[idx]}: {bar} {prob:.1f}%")

    # Tab 2: Upload
    with tab2:
        st.subheader("📁 Upload an Image")
        st.caption("Upload a photo of a plant leaf from your device")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Process and predict
            with st.spinner("🔄 Analyzing..."):
                # Convert to RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Preprocess
                image_array = preprocess_image(image, input_details)

                # Predict
                class_name, confidence, prediction = predict(
                    interpreter, image_array, input_details, output_details, class_names
                )

                # Determine if healthy or diseased
                is_healthy = 'healthy' in class_name.lower()

                # Show result
                if is_healthy:
                    st.success(f"✅ **Diagnosis:** {class_name}")
                else:
                    st.warning(f"⚠️ **Diagnosis:** {class_name}")

                # Show confidence
                st.metric("Confidence", f"{confidence * 100:.1f}%")

                # Show all probabilities
                with st.expander("📊 View all probabilities"):
                    sorted_indices = np.argsort(prediction)[::-1]
                    for idx in sorted_indices[:5]:
                        prob = prediction[idx] * 100
                        bar = "█" * int(prob / 5)
                        st.write(f"{class_names[idx]}: {bar} {prob:.1f}%")

    # Tab 3: About
    with tab3:
        st.subheader("ℹ️ About PlantDoctor")
        st.markdown("""
        ### 🌿 PlantDoctor - AI-Powered Plant Disease Detection

        This app uses a deep learning model trained on the PlantVillage dataset to 
        identify plant diseases from leaf images.

        **Model Details:**
        - Architecture: MobileNetV2 with transfer learning
        - Classes: 15 (Pepper, Potato, Tomato + diseases)
        - Training images: 16,504
        - Validation images: 4,134

        **Supported Plants:**
        - Pepper (Bacterial spot, Healthy)
        - Potato (Early blight, Late blight, Healthy)
        - Tomato (Bacterial spot, Early blight, Healthy, Late blight, 
          Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, 
          Mosaic virus, Yellow Leaf Curl virus)

        **How to Use:**
        1. Take a photo or upload an image of a leaf
        2. The AI will analyze it
        3. Get instant diagnosis with confidence score
        """)


if __name__ == "__main__":
    main()