# project-aol-machine-learning
# PANDU — Academic Prediction & Early Navigation System for Universities

PANDU is a Streamlit-based web application that predicts whether a university student will graduate or drop out based on their academic and demographic data. The application uses three pre-trained Machine Learning models to generate predictions.

---

## Models Used

- Random Forest
- Logistic Regression
- Decision Tree

---

## Features

- Real-time prediction of student status (Graduate / Dropout)
- Side-by-side comparison of all three model performances
- Prediction history saved throughout the session

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/billiefernanda/project-ml.git
cd project-ml/web
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Start the application**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Dataset

The dataset is sourced from Kaggle: [Predict Students' Dropout and Academic Success](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)

It contains student information from a higher education institution, including demographic, academic, and socio-economic attributes. In this application, predictions are focused on two classes: Graduate and Dropout.

---

## Tech Stack

- Python
- Streamlit
- Scikit-Learn
- Pandas, NumPy

---

## Team

| Name | GitHub |
|---|---|
| Billie Fernanda | [@billiefernanda](https://github.com/billiefernanda) |
| Edrico Felician | — |
| Jose Andreas Tandiono | — |
| Felix Dickson Gilianto | — |
