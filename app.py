import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

model = tf.saved_model.load("final_model")

def main():
    st.title("Image Segmentation")

    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    clear_button = st.sidebar.button("Clear")

    if clear_button:
        uploaded_file = None

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Resize and preprocess the image
        image = Image.open(uploaded_file)
        image = image.resize((128, 128))
        image_np = np.array(image)
        image_np = tf.image.resize(image_np, [128, 128])
        image_np = tf.cast(image_np, tf.float32) / 255.0

        pred = model.serve(np.array([image_np]))

        mask = np.argmax(pred, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        mask = tf.image.resize(mask, [image_np.shape[0], image_np.shape[1]])

        mask_image = tf.keras.preprocessing.image.array_to_img(mask[0])

        st.image(mask_image,caption='Segmented Image', use_column_width=True)


if __name__ == "__main__":
    main()

