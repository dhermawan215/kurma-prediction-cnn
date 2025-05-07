import streamlit as st
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image

# Mapping nama file model
model_files = {
    "Model CNN 1": "cnn_full_connected_with_early_stopping.h5",
    "Model CNN 2": "cnn_full_connected_without_early_stopping.h5",
    "Model CNN 3": "kurma_cnn_model_biasa.h5"
}
class_mapping = ['medium-ripe', 'over-ripe', 'ripe','under-ripe','unripe'] 
# UI untuk memilih model
selected_model_name = st.selectbox("Pilih Model", list(model_files.keys()))
selected_model_path = model_files[selected_model_name]

# Load model yang dipilih
@st.cache_resource
def load_model(model_path):
    return keras.models.load_model(model_path)

model = load_model(selected_model_path)

# Preprocessing gambar
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

# Tampilan Streamlit
st.title("Kurma Classification")

uploaded_files = st.file_uploader("Upload Gambar Kurma", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.image(uploaded_file, caption=f"Gambar: {uploaded_file.name}", use_container_width=True)

        img = preprocess_image(uploaded_file)
        pred = model.predict(img)
        predicted_class = np.argmax(pred)
        predicted_class_name = class_mapping[predicted_class]
        confidence = pred[0][predicted_class]

        st.markdown(f"**Model digunakan:** {selected_model_name}")
        st.markdown(f"**Prediksi:** {predicted_class_name} (index: {predicted_class})")
        st.markdown(f"**Probabilitas:** {confidence:.2%}")
        st.markdown(f"**Distribusi Probabilitas:** `{np.round(pred[0], 3).tolist()}`")

