# 🏥 Health Claim Prediction

A machine learning-based web application that predicts whether a healthcare insurance claim is likely to be **Approved or Rejected** based on claim and patient-related information.

## 📌 Project Overview

The project uses **Logistic Regression** to classify healthcare insurance claims.

The trained machine learning model is integrated with a **Streamlit** web application where users can enter claim details and receive:

- Claim approval/rejection prediction
- Prediction probability
- Model-based explanation of the result

## 🎯 Problem Statement

Healthcare insurance companies process a large number of claims. Manual claim evaluation can be time-consuming and may lead to inconsistent decisions.

This project aims to build a machine learning system that can assist in predicting claim outcomes based on historical claim information.

## 🚀 Features

- Patient age input
- Claim amount input
- Previous claims information
- Previous rejection information
- Missing document selection
- Provider risk selection
- Diagnosis selection
- Procedure selection
- Claim approval/rejection prediction
- Prediction probability display
- Custom classification threshold

## 🤖 Machine Learning

### Model Used

**Logistic Regression**

The project also evaluated a Random Forest model and compared the models using multiple evaluation metrics.

### Model Selection

Based on the evaluation performed during the project, Logistic Regression was selected as the final model.

The selected classification threshold is:

**0.35**

A probability greater than or equal to 0.35 is classified as **Claim Approved**, while a probability below 0.35 is classified as **Claim Rejected**.

## 📊 Model Performance

### Logistic Regression

| Metric | Score |
|---|---:|
| CV Mean AUC | 0.677 |
| CV Mean F1 | 0.389 |
| Test AUC | 0.656 |
| Test F1 | 0.556 |
| Test Recall | 0.691 |
| Test Specificity | 0.591 |

### Confusion Matrix

```text
[[78 54]
 [21 47]]