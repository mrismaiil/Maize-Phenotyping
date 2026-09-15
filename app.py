import streamlit as st
import joblib
import cv2
import numpy as np
import pandas as pd
import io
from datetime import datetime

# ==========================================
# LOAD MODELS
# ==========================================
tip_rf = joblib.load('tipshape_rf_new.pkl')
antho_rf = joblib.load('anthocyanin_rf_new.pkl')

# ==========================================
# FEATURE EXTRACTION
# ==========================================
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
            features.extend([
                np.percentile(channel_pixels, 25),
                np.percentile(channel_pixels, 50),
                np.percentile(channel_pixels, 75)
            ])
        else:
            features.extend([0, 0, 0])
    for i in range(3):
        hist = cv2.calcHist([img], [i], mask, [8], [0, 256]).flatten()
        total = np.sum(hist)
        features.extend(hist / total if total > 0 else hist)
    features.append(lab_leaf[1])
    features = np.pad(features, (0, max(0, 50 - len(features))), 'constant', constant_values=0)
    return features[:50]

# ==========================================
# APP LAYOUT
# ==========================================
st.set_page_config(page_title="Maize Phenotyping AI", page_icon="🌽", layout="wide")
st.title("🌽 Maize Phenotyping AI")
st.write("Upload multiple maize seedling images to get Tip Shape (1–5) and Anthocyanin (1–9) grades.")

# Multi-file uploader
uploaded_files = st.file_uploader(
    "Upload images (you can select multiple files)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"**{len(uploaded_files)} image(s) uploaded**")
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.write(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})...")
        
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Extract features
        features = extract_features(img).reshape(1, -1)
        
        # Predictions
        tip_pred = int(tip_rf.predict(features)[0])
        tip_conf = float(np.max(tip_rf.predict_proba(features)[0]) * 100)
        
        antho_pred = int(antho_rf.predict(features)[0])
        antho_conf = float(np.max(antho_rf.predict_proba(features)[0]) * 100)
        
        # Store result
        results.append({
            "Image Name": uploaded_file.name,
            "Tip Shape Grade": tip_pred,
            "Tip Shape Confidence (%)": round(tip_conf, 2),
            "Anthocyanin Grade": antho_pred,
            "Anthocyanin Confidence (%)": round(antho_conf, 2)
        })
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.write("✅ All images processed!")
    progress_bar.empty()
    
    # ==========================================
    # DISPLAY RESULTS AS EDITABLE TABLE
    # ==========================================
    st.subheader("📊 Results")
    st.write("You can edit the grades directly in the table below before exporting.")
    
    df = pd.DataFrame(results)
    
    # Editable table
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Tip Shape Grade": st.column_config.NumberColumn(min_value=1, max_value=5, step=1),
            "Anthocyanin Grade": st.column_config.NumberColumn(min_value=1, max_value=9, step=1),
        }
    )
    
    # ==========================================
    # EXPORT OPTIONS
    # ==========================================
    st.subheader("💾 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"maize_phenotyping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            edited_df.to_excel(writer, index=False, sheet_name='Results')
        excel_data = excel_buffer.getvalue()
        st.download_button(
            label="📥 Download as Excel (.xlsx)",
            data=excel_data,
            file_name=f"maize_phenotyping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    # ==========================================
    # IMAGE PREVIEW (expandable)
    # ==========================================
    with st.expander("🖼️ View Uploaded Images"):
        cols = st.columns(3)
        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i % 3]:
                st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
