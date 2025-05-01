import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetv2_preprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
import matplotlib.pyplot as plt
import os

# --- Setup Paths ---
current_dir = os.path.dirname(os.path.abspath(__file__))  # Code/Trashnet/ui/
code_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))  # Code/
MODEL_DIR = os.path.join(code_dir, "Trashnet", "code")  # Code/Trashnet/code/


# --- Constants ---
IMG_SIZE = (224, 224)
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
MODEL_CONFIGS = {
    'EfficientNetV2S': efficientnetv2_preprocess,
    'MobileNet': mobilenet_preprocess,
    'MobileNetV2': mobilenetv2_preprocess,
    'MobileNetV3Small': mobilenetv2_preprocess,
    'MobileNetV3Large': mobilenetv2_preprocess,
    'ResNet101V2': resnet_preprocess,
    'ResNet152V2': resnet_preprocess
}

# --- Helper Functions ---
def load_model_and_history(model_name):
    """
    Load the selected model and its training history.
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}_trashnet_finetuned.h5")
    history_path = os.path.join(MODEL_DIR, f"{model_name}_finetuned_history.npy")

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model {model_name}: {str(e)}")
        return None, None

    try:
        history = np.load(history_path, allow_pickle=True).item()
    except Exception as e:
        st.warning(f"Could not load history for {model_name}: {str(e)}")
        history = None

    return model, history

def preprocess_image(image, preprocess_fn):
    """
    Preprocess the uploaded image to match the training setup.
    """
    # Resize to 224x224
    image = image.resize(IMG_SIZE)
    # Convert to array
    image_array = img_to_array(image)
    # Expand dimensions to match model input (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    # Apply model-specific preprocessing
    image_array = preprocess_fn(image_array)
    return image_array

def plot_confidence_scores(probabilities, classes):
    """
    Plot a bar chart of confidence scores for each class.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(classes, probabilities, color='skyblue')
    ax.set_xlabel("Trash Type")
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Confidence Scores")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# --- Title ---
st.title("Trash Classification App")

# --- Model Selection ---
model_choice = st.selectbox(
    "Choose a model:",
    list(MODEL_CONFIGS.keys())
)

# --- Load Model and History ---
model, history = load_model_and_history(model_choice)

if model is None:
    st.stop()  # Stop execution if model loading fails

# --- Display Model Performance Summary ---
if history is not None:
    st.subheader(f"Performance Summary for {model_choice}")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Final Training Accuracy: {history['accuracy'][-1]:.4f}")
        st.write(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
    with col2:
        st.write(f"Final Training Loss: {history['loss'][-1]:.4f}")
        st.write(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")

    # --- Display Training Curves ---
    training_curves_path = os.path.join(MODEL_DIR, f"training_curves_{model_choice}.png")
    if os.path.exists(training_curves_path):
        st.subheader("Training Curves")
        st.image(training_curves_path, caption=f"Training Curves for {model_choice}", use_column_width=True)

# --- Upload Image ---
uploaded_file = st.file_uploader("Upload a trash image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# --- Predict Button ---
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')  # Ensure image is in RGB format
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            # --- Preprocess Image ---
            preprocess_fn = MODEL_CONFIGS[model_choice]
            img_array = preprocess_image(img, preprocess_fn)

            # --- Make Prediction ---
            probabilities = model.predict(img_array)[0]
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = CLASSES[predicted_class_idx]
            confidence = probabilities[predicted_class_idx]

            # --- Display Prediction ---
            st.success(f"Predicted Trash Type: **{predicted_class}**")
            st.info(f"Confidence: **{confidence:.2f}**")

            # --- Display Confidence Scores for All Classes ---
            st.subheader("Confidence Scores for All Classes")
            fig = plot_confidence_scores(probabilities, CLASSES)
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing the image or making prediction: {str(e)}")