# 🏥 HealthClaim-AI

### Machine Learning-Based Health Insurance Claim Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Bhuvanabodapati/HealthClaim-AI)

> An end-to-end machine learning application that predicts whether a healthcare insurance claim is likely to be approved or rejected based on claim and patient-related attributes.

### 🚀 Live Demo

**[Open HealthClaim-AI →](https://healthclaim-ai.streamlit.app)**

### 💻 Source Code

**[GitHub Repository →](https://github.com/Bhuvanabodapati/HealthClaim-AI)**

---

## 📌 Project Overview

HealthClaim-AI is a machine learning-based healthcare insurance claim prediction system designed to assist in the preliminary assessment of insurance claims.

The project follows an end-to-end ML workflow:

**Data → Exploration → Preprocessing → Model Training → Model Evaluation → Threshold Selection → Model Serialization → Streamlit Deployment**

The application provides an interactive interface where users can enter claim information and receive:

- Claim approval/rejection prediction
- Prediction probability
- Simple interpretation of the model output

The system is intended as an **educational machine learning demonstration** and is not designed for real-world insurance or medical decision-making.

---

## 🎯 Problem Statement

Insurance claim processing can involve reviewing multiple factors such as claim amount, previous claims, previous rejections, missing documentation, provider risk, diagnosis, and procedure type.

The objective of this project is to build a machine learning classification system that can learn patterns from historical claim data and predict the likely outcome of a new claim.

### Objective

Build a classification model that:

1. Processes structured healthcare claim data
2. Handles numerical and categorical features
3. Learns patterns associated with claim outcomes
4. Produces a probability-based prediction
5. Provides predictions through an interactive web application

---

## ✨ Key Features

- 🧠 **Machine Learning Classification**
  - Logistic Regression used as the final selected model

- 📊 **Model Evaluation**
  - Cross-validation
  - ROC-AUC
  - F1 Score
  - Recall / Sensitivity
  - Specificity
  - Confusion Matrix

- 🎚️ **Decision Threshold Optimization**
  - A classification threshold of **0.35** was selected to improve the balance between recall and other classification metrics.

- 🔄 **Preprocessing Pipeline**
  - Numerical and categorical feature preprocessing
  - Saved preprocessing object for consistent inference

- 💾 **Model Serialization**
  - Trained model and preprocessing pipeline are stored using `joblib`

- 🖥️ **Interactive Streamlit Application**
  - User-friendly claim input interface
  - Real-time prediction
  - Prediction probability
  - Human-readable result

- ☁️ **Cloud Deployment**
  - Deployed using Streamlit Community Cloud

---

## 🧠 Machine Learning Approach

### 1. Data Preparation

The healthcare claims dataset is loaded and prepared for machine learning.

The workflow includes:

- Data inspection
- Feature selection
- Missing-value handling
- Numerical feature preprocessing
- Categorical feature encoding
- Train-test splitting

### 2. Models Evaluated

Two classification models were evaluated:

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline classification model |
| Random Forest | Non-linear ensemble comparison model |

### 3. Model Selection

Logistic Regression was selected as the final model based on cross-validation performance and its stronger recall/F1 trade-off for this project.

---

## 📊 Model Performance

### Final Model: Logistic Regression

**Classification Threshold:** `0.35`

| Metric | Score |
|---|---:|
| CV Mean AUC | **0.677** |
| CV Mean F1 | **0.389** |
| Test AUC | **0.656** |
| Test F1 | **0.556** |
| Test Recall | **0.691** |
| Test Specificity | **0.591** |

### Confusion Matrix

```text
[[78 54]
 [21 47]]