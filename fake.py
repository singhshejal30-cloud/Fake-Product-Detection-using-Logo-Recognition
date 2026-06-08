# =========================================================
# FAKE PRODUCT DETECTION USING LOGO RECOGNITION
# FINAL WORKING STREAMLIT PROJECT
# =========================================================

# INSTALL REQUIRED LIBRARIES:
# pip install streamlit ultralytics opencv-python matplotlib pillow numpy

# RUN PROJECT:
# streamlit run fake.py

# =========================================================

import streamlit as st
import cv2
import os
import numpy as np
from ultralytics import YOLO

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fake Logo Detection",
    page_icon="🛍️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background-image: url("https://images.unsplash.com/photo-1523275335684-37898b6baf30");
    background-size: cover;
    background-attachment: fixed;
}

/* Main Title */

.main-title {
    font-size: 55px;
    font-weight: bold;
    color: white;
    text-align: center;
    animation: fadeIn 2s;
}

/* Subtitle */

.sub-title {
    font-size: 24px;
    color: white;
    text-align: center;
    margin-bottom: 40px;
}

/* Result Box */

.result-box {
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}

/* Real Product */

.real {
    background-color: rgba(0, 128, 0, 0.8);
}

/* Fake Product */

.fake {
    background-color: rgba(255, 0, 0, 0.8);
}

/* Animation */

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">Fake Product Detection using Logo Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">OpenCV + YOLOv8 + SIFT Feature Matching</div>',
    unsafe_allow_html=True
)

# =========================================================
# LOAD YOLO MODEL
# =========================================================

model = YOLO("yolov8n.pt")

# =========================================================
# DATASET PATH
# =========================================================

# CHANGE THIS PATH ACCORDING TO YOUR COMPUTER

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "fake_logo")

# =========================================================
# LOAD LOGO DATASET
# =========================================================

logos = {}

brands = os.listdir(dataset_path)

for brand in brands:

    brand_path = os.path.join(dataset_path, brand)

    logos[brand] = []

    files = os.listdir(brand_path)

    for file in files:

        path = os.path.join(brand_path, file)

        img = cv2.imread(path, 0)

        if img is not None:

            logos[brand].append(img)

# =========================================================
# SIFT DETECTOR
# =========================================================

sift = cv2.SIFT_create()

# =========================================================
# IMAGE COMPARISON FUNCTION
# =========================================================

def compare_images(img1, img2):

    # Resize images
    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))

    # Detect features
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # If no descriptors
    if des1 is None or des2 is None:
        return 0

    # Matcher
    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    good = []

    # Lowe ratio test
    for m, n in matches:

        if m.distance < 0.75 * n.distance:
            good.append(m)

    return len(good)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("About Project")

st.sidebar.info("""
This project detects fake branded logos using:

✅ OpenCV  
✅ YOLOv8  
✅ SIFT Feature Matching  
✅ Streamlit  

Upload a logo image to detect whether it is REAL or FAKE.
""")

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload Logo Image",
    type=["png", "jpg", "jpeg"]
)

# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    test_img_color = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    # Check image
    if test_img_color is None:

        st.error("Image not loaded properly")

    else:

        # =================================================
        # SHOW UPLOADED IMAGE
        # =================================================

        st.image(
            cv2.cvtColor(test_img_color, cv2.COLOR_BGR2RGB),
            caption="Uploaded Logo",
            use_container_width=True
        )

        # =================================================
        # YOLO DETECTION
        # =================================================

        results = model(test_img_color)

        detected_image = results[0].plot()

        st.image(
            detected_image,
            caption="YOLO Detection",
            use_container_width=True
        )

        # =================================================
        # CONVERT TO GRAYSCALE
        # =================================================

        test_img = cv2.cvtColor(
            test_img_color,
            cv2.COLOR_BGR2GRAY
        )

        # =================================================
        # BRAND MATCHING
        # =================================================

        st.write("## Brand Matching Scores")

        best_brand = ""
        best_score = 0

        for brand, images in logos.items():

            total = 0

            for logo in images:

                score = compare_images(
                    test_img,
                    logo
                )

                total += score

            avg_score = total / len(images)

            st.write(
                f"### {brand} : {avg_score:.2f}"
            )

            # Best score
            if avg_score > best_score:

                best_score = avg_score
                best_brand = brand

        # =================================================
        # FINAL RESULT
        # =================================================

        st.write("## Final Detection")

        st.write(
            f"### Detected Brand : {best_brand}"
        )

        st.write(
            f"### Matching Score : {best_score:.2f}"
        )

        # =================================================
        # REAL / FAKE PREDICTION
        # =================================================

        if best_score > 15:

            st.markdown(
                '''
                <div class="result-box real">
                ✅ REAL LOGO DETECTED
                </div>
                ''',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '''
                <div class="result-box fake">
                ❌ FAKE LOGO DETECTED
                </div>
                ''',
                unsafe_allow_html=True
            )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <h4 style='color:white;'>
    Developed using Streamlit | OpenCV | YOLOv8 | SIFT
    </h4>
    </center>
    """,
    unsafe_allow_html=True
)
