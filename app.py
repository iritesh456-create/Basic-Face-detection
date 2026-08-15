import streamlit as st
import cv2
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(page_title="Face Detection", page_icon="👤", layout="centered")

# --- 2. Load and Cache the Detector ---
# Caching ensures the XML file is loaded once into memory instead of on every rerun
@st.cache_resource
def load_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)

face_detector = load_detector()

# --- 3. Build the UI ---
st.title("👤 Face Detection App")
st.write("Upload a photo or snap one live with your camera.")

# Dual input choice
source = st.radio("Choose Input Method:", ["Upload photo", "Use camera"], horizontal=True)

if source == "Upload photo":
    image_file = st.file_uploader("Choose a photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("Take a photo")

# --- 4. Process and Detect ---
if image_file is not None:
    # Read bytes directly and decode into an OpenCV image (BGR)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Convert to grayscale for Haar Cascade[cite: 1]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Run detection[cite: 1]
    faces = face_detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw bounding boxes on a copy of the image[cite: 1]
    result = image.copy()
    for (x, y, width, height) in faces:
        # (0, 255, 0) gives a crisp green bounding box in BGR
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 3)

    # Convert BGR back to RGB for correct display in Streamlit[cite: 1]
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    st.divider()
    st.image(result_rgb, caption=f"Detected Faces: {len(faces)}", use_container_width=True)

    # Status alerts
    if len(faces) == 0:
        st.warning("No face found. Try a clear, front-facing photo.")
    else:
        st.success(f"Successfully detected {len(faces)} face(s)!")

st.caption("Built with OpenCV Haar Cascades & Streamlit")