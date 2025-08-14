# Breast_Cancer_Detection
# 🎗️ Breast Cancer Detection using Machine Learning

This project applies **Machine Learning** techniques to detect breast cancer based on patient diagnostic data.  
Using the **Wisconsin Breast Cancer Dataset**, the goal is to classify tumors as **Malignant (M)** or **Benign (B)**, helping in early detection and diagnosis.

---

## 📂 Dataset Overview

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Samples:** 569
- **Features:** 30 numerical features derived from cell nucleus measurements in images.
- **Target Classes:**
  - 🟥 Malignant — Cancerous tumor
  - 🟦 Benign — Non-cancerous tumor

---

## 🛠 Workflow

1. **📊 Exploratory Data Analysis (EDA)**
   - Data structure inspection
   - Missing value analysis
   - Feature distribution & correlations
   - Visualization using histograms, pairplots, heatmaps

2. **🧹 Data Preprocessing**
   - Handling categorical variables (`diagnosis`: M/B)
   - Encoding target variable (M=1, B=0)
   - Feature scaling using StandardScaler
   - Train-test split (80/20)

3. **🤖 Model Training**
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Decision Tree Classifier
   - Random Forest Classifier
   - Support Vector Machine (SVM)

4. **📈 Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - ROC Curve & AUC Score

---

## 📦 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
Main Libraries Used:

pandas

numpy

matplotlib

seaborn

scikit-learn

🚀 How to Run
Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/breast-cancer-detection-ml.git
cd breast-cancer-detection-ml
Run the script / notebook

bash
Copy
Edit
python breast_cancer_detection.py
or open the Jupyter notebook:

bash
Copy
Edit
jupyter notebook breast_cancer_detection.ipynb

📊 Example Results
Confusion Matrix:
Predicted Malignant	Predicted Benign
Actual Malignant	70	2
Actual Benign	1	101
ROC Curve:
AUC Score: ~0.99 ✅

🔍 Insights from EDA
Mean radius, texture, and concavity are highly correlated with malignancy.
Dataset is balanced enough to train models without heavy class imbalance handling.
Strong separation between classes in PCA-transformed space.
