import streamlit as st
import numpy as np
import os
import requests
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image
from io import BytesIO

def run():
    # Initiate Class name
    CLASS_NAMES = ['Alluvial_Soil', 'Arid_Soil', 'Black_Soil', 'Laterite_Soil', 'Mountain_Soil', 'Red_Soil', 'Yellow_Soil']
    IMG_SIZE = 224



    # Load model from google drive
    @st.cache_resource
    def load_model():
        model_path = "model_MobileNetV2.h5"

        if not os.path.exists(model_path):
            with st.spinner("Downloading model..."):
                file_id = "1NxuFM_s5OFl71sxCA5_52oKss_lnVQaQ"
                url = f"https://drive.google.com/uc?id={file_id}"

                gdown.download(url, model_path, quiet=False)

        st.write("Model exists:", os.path.exists(model_path))
        st.write("Model size:", os.path.getsize(model_path))

        return keras_load_model(model_path)
    
# https://drive.google.com/file/d/1NxuFM_s5OFl71sxCA5_52oKss_lnVQaQ/view?usp=sharing

    model = load_model()

    #  Function to predict using model
    def prediction(img: Image.Image):
        img_resized = img.resize((IMG_SIZE, IMG_SIZE))
        x = tf.keras.utils.img_to_array(img_resized) / 255.0
        x = np.expand_dims(x, axis=0)
        proba = model.predict(x, batch_size=1)[0]
        pred_idx = np.argmax(proba)
        return CLASS_NAMES[pred_idx], float(proba[pred_idx]) * 100, proba

    st.markdown("## ⛰️ Soil Image Identification")
    st.caption("Identify soil from images using deep learning (MobileNetV2)")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    with col2:
        img_url = st.text_input("Or paste image URL")
        predict_url = st.button("Predict from URL", use_container_width=True)

    img = None

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')

    elif predict_url and img_url:
        try:
            response = requests.get(img_url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            st.error(f"Gagal mengambil gambar dari URL: {e}")

    if img is not None:
        st.divider()
        col_img, col_result = st.columns([1, 1])

        with col_img:
            st.image(img, caption="Input image", use_column_width=True)

        with col_result:
            with st.spinner("Analyzing..."):
                label, confidence, proba = prediction(img)

            # Final result
            st.metric(label="Predicted class", value=label.capitalize())

            # Badge confidence
            color = "green" if confidence >= 70 else "orange" if confidence >= 40 else "red"
            st.markdown(
                f"<span style='background:#E1F5EE;color:#085041;padding:4px 12px;"
                f"border-radius:999px;font-size:13px'>Confidence: {confidence:.1f}%</span>",
                unsafe_allow_html=True
            )

            st.markdown("**All class probabilities**")

            # Progress bar per kelas
            sorted_idx = np.argsort(proba)[::-1]
            for i in sorted_idx:
                pct = float(proba[i]) * 100
                bar_color = "#1D9E75" if i == np.argmax(proba) else "#AFA9EC"
                st.markdown(
                    f"""
                    <div style='margin-bottom:6px'>
                    <div style='display:flex;justify-content:space-between;font-size:12px;
                                color:var(--text-color);margin-bottom:2px'>
                        <span>{CLASS_NAMES[i]}</span><span>{pct:.1f}%</span>
                    </div>
                    <div style='background:#eee;border-radius:999px;height:6px'>
                        <div style='width:{pct:.1f}%;background:{bar_color};
                                    border-radius:999px;height:6px'></div>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )