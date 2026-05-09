<div align="center">

# 🩺 DiabetesScan AI

### Advanced Diabetes Risk Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

*A futuristic, AI-powered medical dashboard that predicts diabetes risk using machine learning — built with a Dark Glassmorphism UI.*

</div>

---

## ✨ Features

- **🧬 Instant Risk Prediction** — Enter clinical values and get a High Risk / Low Risk result in seconds
- **⚖️ Smart BMI Calculator** — Don't know your BMI? Toggle on weight + height and it's computed automatically
- **🎨 Dark Glassmorphism UI** — Deep navy background, frosted glass cards, neon accents, and animated result cards
- **📊 Model Transparency** — View feature importance rankings, dataset stats, and model performance metrics
- **💡 Health Tips** — Tabbed nutrition, exercise, and lifestyle guidance to help users act on their results
- **🔄 Sidebar Navigation** — Clean 3-page layout: Prediction Engine | About & Model | Health Tips
- **📋 Input Summary** — After prediction, review all entered values against normal medical reference ranges

---

## 🖥️ Screenshots

| Prediction Engine | Result Cards |
|---|---|
| Dark glassmorphism input form with 3 grouped cards | Animated glowing High Risk / Low Risk status card |

> *The app features a real-time BMI toggle, neon hover buttons, and live probability percentages.*

---

## 🗂️ Project Structure

```
DiabetesScan-AI/
│
├── app.py                  # Main Streamlit application
├── best_model.pkl          # Trained RandomForest model (joblib)
├── scaler.pkl              # Fitted StandardScaler (joblib)
├── diabetes.csv            # Pima Indians Diabetes dataset
├── diabetes_ml_project.py  # Full ML pipeline (EDA → training → export)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DiabetesScan-AI.git
cd DiabetesScan-AI
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 🤖 Model Details

| Property | Value |
|---|---|
| **Algorithm** | Random Forest Classifier |
| **Estimators** | 100 Decision Trees |
| **Accuracy** | 73.4% |
| **ROC-AUC** | 0.810 |
| **Feature Scaling** | StandardScaler |
| **Class Imbalance** | Random Oversampling on training set |
| **Missing Values** | Median imputation (glucose, BP, insulin, skin thickness, BMI) |
| **Outlier Treatment** | IQR Winsorization |

### Input Features

| Feature | Unit | Description |
|---|---|---|
| Glucose | mg/dL | Plasma glucose concentration (2-hour OGTT) |
| Blood Pressure | mm Hg | Diastolic blood pressure |
| Skin Thickness | mm | Triceps skin fold thickness |
| Insulin | µU/mL | 2-hour serum insulin |
| BMI | kg/m² | Body Mass Index |
| Diabetes Pedigree Function | — | Genetic risk score based on family history |
| Age | years | Patient age |
| Pregnancies | count | Number of times pregnant |

---

## 📊 Dataset

This project uses the **Pima Indians Diabetes Dataset**, originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

- **768** patient records
- **8** clinical features + 1 binary target (`Outcome`)
- **Class distribution:** 65% negative (no diabetes) / 35% positive (diabetes)
- Publicly available on [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and the [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes)

---

## 🔁 Re-Training the Model

To retrain the model from scratch using your own data or updated parameters, run:

```bash
python diabetes_ml_project.py
```

This script covers the full ML pipeline:
1. Data loading & understanding
2. Cleaning (duplicates, zero imputation, outliers)
3. Exploratory Data Analysis (EDA) with plots
4. Preprocessing (scaling, encoding, oversampling)
5. Clustering (KMeans)
6. Classification (Logistic Regression vs Random Forest)
7. Model export → `best_model.pkl`

> **Note:** Output plots and the model file are saved to an `./outputs/` directory.

---

## 🚀 Deploying to Streamlit Cloud

1. Push your repository to GitHub (make sure `app.py`, `best_model.pkl`, `scaler.pkl`, and `requirements.txt` are all committed)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — your app will be live in under a minute

---

## ⚠️ Disclaimer

This application is intended for **educational and informational purposes only**. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified physician or healthcare provider regarding any medical condition. Never disregard professional medical advice based on the output of this tool.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) — feel free to use, modify, and distribute it with attribution.

---

## 🙏 Acknowledgements

- **Dataset:** Smith, J.W. et al. — *Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus* (1988)
- **UI Inspiration:** Dark glassmorphism design patterns from modern SaaS dashboards
- **ML Framework:** [scikit-learn](https://scikit-learn.org/) by the sklearn contributors
- **App Framework:** [Streamlit](https://streamlit.io/) by Snowflake

---

<div align="center">
  Made with ❤️ and 🩺 &nbsp;|&nbsp; <em>For educational use only</em>
</div>
