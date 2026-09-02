# 🌽 Maize Seedling Phenotyping AI

An automated image-based tool for grading two key DUS (Distinctness, Uniformity, Stability) descriptors in maize seedlings.

## Phenotyping Traits

| Trait | Scale | Description |
|-------|-------|-------------|
| **First leaf tip shape** | Grades 1–5 | Visual classification of leaf tip morphology |
| **Anthocyanin coloration of the first leaf sheath** | Grades 1–9 | Intensity of purple pigmentation on the leaf sheath |

## How It Works

1. Upload a high-resolution photo of the first leaf sheath.
2. The system extracts color and shape features (RGB, LAB, HSV, contour geometry).
3. A trained Random Forest classifier predicts the grades for both traits.
4. Confidence scores are displayed alongside each prediction.

## Model Performance

| Trait | AI1 (exact match) | AI2 (within 1 grade) |
|-------|-------------------|----------------------|
| First leaf tip shape | 52.38% | 90.48% |
| Anthocyanin coloration of the first leaf sheath | 24.76% | 60.95% |

- **AI1:** Percentage of exact grade matches.
- **AI2:** Percentage of predictions within one grade of the true label (the standard "1-code difference" acceptance criterion used in DUS testing).

## Dataset

- **3,027** maize seedling images
- Standardized imaging conditions (consistent lighting, neutral gray background)
- Expert-assigned grades following GB/T 19557.24-2018 guidelines

## Usage

```bash
pip install -r requirements.txt
streamlit run app.py
