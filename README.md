# Customer Churn Prediction using Machine Learning

## Overview

This project implements a machine learning solution for predicting whether a customer is likely to churn.

The project was developed as part of an AI/ML machine test.

The solution covers:

- Data understanding
- Data cleaning
- Missing-value handling
- Categorical encoding
- Feature scaling
- Machine learning model training
- Model evaluation
- REST API deployment
- GitHub documentation

---

## Problem Statement

Customer churn prediction helps businesses identify customers who may stop using their services.

The objective of this project is to predict the churn status of a customer using customer demographic, service, contract, and billing information.

---

## Dataset

The dataset is stored in:

`data/customer_churn.csv`

The dataset contains both numerical and categorical features.

Examples of numerical features:

- SeniorCitizen
- tenure
- MonthlyCharges
- TotalCharges

Examples of categorical features:

- gender
- Partner
- Dependents
- PhoneService
- InternetService
- Contract
- PaymentMethod

The target variable is:

`Churn`

where:

- Yes = customer churned
- No = customer did not churn

---

## Data Preprocessing

The following preprocessing steps were implemented:

### Missing Values

Missing numerical values are replaced using the median.

Missing categorical values are replaced using the most frequent category.

### Categorical Encoding

Categorical variables are converted into numerical representations using One-Hot Encoding.

### Feature Scaling

Numerical features are standardized using StandardScaler.

This is especially useful for Logistic Regression because features with different scales can affect model optimization.

---

## Machine Learning Models

Two classification algorithms were implemented.

### 1. Logistic Regression

Logistic Regression was selected as a baseline classification model.

It is simple, interpretable, and effective for binary classification problems.

### 2. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

It can capture nonlinear relationships between features and the target variable.

---

## Model Evaluation

Both models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

The F1 score was used to select the final model because it balances precision and recall.

The training script automatically compares the two models and saves the model with the better F1 score.

---

## Feature Impact

Important factors that can influence customer churn include:

- Contract type
- Tenure
- Monthly charges
- Total charges
- Internet service
- Payment method
- Customer demographics

For example, customers with month-to-month contracts may have different churn behavior compared with customers with longer-term contracts.

The actual feature impact depends on the trained model and dataset.

---

## API

A FastAPI application was created for deployment.

Start the API using:

```bash
uvicorn api:app --reload