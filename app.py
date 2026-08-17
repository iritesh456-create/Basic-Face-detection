import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Detection", page_icon="👤")
st.title("👤 Face Detection")
st.write("Detect human faces using OpenCV Haar Cascade.")

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    st.error("Could not load the Haar Cascade face detector.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image containing one or more faces",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    face_image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if face_image is None:
        st.error("Could not read the uploaded image.")
        st.stop()

    gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    output_image = face_image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    st.success(f"Faces detected: {len(faces)}")
    st.image(output_rgb, caption=f"Detected Faces: {len(faces)}", use_container_width=True)
else:
    st.info("Upload a photo to begin.")
