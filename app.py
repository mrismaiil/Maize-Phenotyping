import streamlit as st
import joblib
import cv2
import numpy as np

tip_rf = joblib.load('tipshape_rf_new.pkl')
antho_rf = joblib.load('anthocyanin_rf_new.pkl')

def extract_features(img_array):
    img = cv2.resize(img_array, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(50)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = w / h if h != 0 else 0
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else 0
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest], -1, 255, -1)
    leaf_rgb = cv2.mean(img, mask=mask)[:3]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv_leaf = cv2.mean(hsv, mask=mask)[:3]
    lab_leaf = cv2.mean(lab, mask=mask)[:3]
    features = []
    features.extend([area, perimeter, circularity, aspect_ratio, solidity, len(contours)])
    features.extend(leaf_rgb)
    features.extend(hsv_leaf)
    features.extend(lab_leaf)
    for i in range(3):
        channel_pixels = img[:, :, i][mask == 255]
        if len(channel_pixels) > 0:
            features.extend([np.percentile(channel_pixels, 25), np.percentile(channel_pixels, 50), np.percentile(channel_pixels, 75)])
        else:
            features.extend([0, 0, 0])
    for i in range(3):
        hist = cv2.calcHist([img], [i], mask, [8], [0, 256]).flatten()
        total = np.sum(hist)
        features.extend(hist / total if total > 0 else hist)
    features.append(lab_leaf[1])
    features = np.pad(features, (0, max(0, 50 - len(features))), 'constant', constant_values=0)
    return features[:50]

st.title("🌽 Maize Phenotyping AI")
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    features = extract_features(img).reshape(1, -1)
    tip_pred = tip_rf.predict(features)[0]
    tip_probs = tip_rf.predict_proba(features)[0]
    tip_conf = np.max(tip_probs) * 100
    antho_pred = antho_rf.predict(features)[0]
    antho_probs = antho_rf.predict_proba(features)[0]
    antho_conf = np.max(antho_probs) * 100
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tip Shape Grade", f"{tip_pred}", f"Confidence: {tip_conf:.1f}%")
    with col2:
        st.metric("Anthocyanin Grade", f"{antho_pred}", f"Confidence: {antho_conf:.1f}%")
    st.image(uploaded_file, caption="Uploaded Image", width=300)
