import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt

@st.cache_resource
def load_models():
    brain_breast_classifier = load_model("brain_vs_breast_classifier.keras")  
    brain_tumor_detector = load_model("Brain_Classification_Model.h5")        
    breast_tumor_detector = load_model("Breast_Classification_Model (1).h5")  
    brain_segmentor = load_model("final_brain_unet_model.h5")                 
    breast_segmentor = load_model("Breast_unet_model.h5")                     
    return brain_breast_classifier, brain_tumor_detector, breast_tumor_detector, brain_segmentor, breast_segmentor

def preprocess_grayscale(img):
    img = img.resize((224, 224)).convert('L')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return np.expand_dims(img_array, axis=0)

def preprocess_rgb(img, size=(224, 224)):
    img = img.resize(size).convert('RGB')
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# segmentation
def show_segmentation_result(original, mask, size=(224, 224)):
    predicted_mask = mask[0, :, :, 0]
    binary_mask = (predicted_mask > 0.5).astype(np.uint8) * 255

    original_resized = np.array(original.resize(size))
    if original_resized.ndim == 2 or original_resized.shape[2] == 1:
        original_resized = cv2.cvtColor(original_resized, cv2.COLOR_GRAY2RGB)

    mask_colored = cv2.applyColorMap(binary_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_resized, 0.6, mask_colored, 0.4, 0)

    
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(original_resized)
    ax[0].set_title("Input Image")
    ax[0].axis("off")

    ax[1].imshow(predicted_mask, cmap='gray')
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")

    ax[2].imshow(overlay)
    ax[2].set_title("Overlay")
    ax[2].axis("off")

    st.pyplot(fig)

st.title("🧠🩺 Brain/Breast Tumor Classification, Detection & Segmentation")

uploaded_file = st.file_uploader("📤 Upload a brain or breast scan image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    st.write("⏳ Processing...")

    bb_classifier, brain_detector, breast_detector, brain_segmentor, breast_segmentor = load_models()

    #Classify brain or breast
    image_for_classification = preprocess_grayscale(image)
    classification = np.argmax(bb_classifier.predict(image_for_classification))

    if classification == 0:
        st.subheader("🧠 Classification: Brain")

        #Brain tumor detection
        image_for_detection = preprocess_rgb(image)
        tumor_prob = brain_detector.predict(image_for_detection)[0][0]
        st.write(f"**Tumor Probability:** `{1 - tumor_prob:.2f}`")

        if tumor_prob < 0.5:
            st.error("🔴 Brain tumor detected")
            if st.checkbox("🧪 Run Brain Tumor Segmentation"):
                input_size = brain_segmentor.input_shape[1:3]
                image_for_segment = preprocess_rgb(image, size=input_size)
                mask = brain_segmentor.predict(image_for_segment)
                show_segmentation_result(image, mask, size=input_size)
        else:
            st.success("🟢 No brain tumor detected")

    else:
        st.subheader("🧬 Classification: Breast")

        #Breast tumor detection
        image_for_detection = preprocess_rgb(image)
        prediction = breast_detector.predict(image_for_detection)
        predicted_class = np.argmax(prediction)
        class_names = ['🟡 Benign', '🔴 Malignant', '🟢 Normal']
        st.write(f"**Tumor Type:** {class_names[predicted_class]}")

        if predicted_class in [0, 1]:
            if st.checkbox("🧪 Run Breast Tumor Segmentation"):
                input_size = breast_segmentor.input_shape[1:3]
                image_for_segment = preprocess_rgb(image, size=input_size)
                mask = breast_segmentor.predict(image_for_segment)
                show_segmentation_result(image, mask, size=input_size)
