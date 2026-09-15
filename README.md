# 🌽 Maize Phenotyping AI

An automated image-based tool for grading two key DUS (Distinctness, Uniformity, Stability) descriptors in maize seedlings:

- **First leaf tip shape** (Grades 1–5)
- **Anthocyanin coloration of the first leaf sheath** (Grades 1–9)

## 🚀 Live Tool

Access the web app here: [maize-phenotyping-mrismail.streamlit.app](https://maize-phenotyping-mrismail.streamlit.app)

## 📖 Overview

This tool uses a **Random Forest classifier** trained on **50 engineered features** (shape, color, texture) extracted from high-resolution maize seedling images.

To address class imbalance, **SMOTE** (Synthetic Minority Over-sampling Technique) was applied during training.

## 📊 Model Performance

Performance is evaluated using two metrics:

- **AI1**: Exact match between model prediction and manual grade
- **AI2**: Prediction within ±1 grade of the manual grade (the standard "1-code difference" acceptance criterion for DUS)

| Trait | AI1 (Exact) | AI2 (Within 1) |
|-------|-------------|----------------|
| First leaf tip shape (1–5) | 78.75% | 95.05% |
| Anthocyanin coloration (1–9) | 62.46% | 81.95% |

## 📁 Dataset

- **3,027** maize seedling images from DUS trials (2015–2023)
- Graded by expert testers following **GB/T 19557.24-2018** guidelines

## 🛠️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
