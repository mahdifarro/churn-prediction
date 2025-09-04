[![CI Pipeline](https://github.com/mahdifarro/churn-prediction/actions/workflows/ci.yaml/badge.svg)](https://github.com/mahdifarro/churn-prediction/actions/workflows/ci.yaml)
## 📊 Dataset: Telco Customer Churn

### Overview

This dataset provides information about customers from a fictional telecommunications company. It includes customer demographics, account information, subscribed services, and whether they have churned or not. The goal is to predict customer churn based on these features.

- **Rows:** 7,043
- **Columns:** 21

### Source

The data is publicly available and was originally provided by IBM as part of their Watson Analytics offerings.

- **Source:** [IBM Telco Customer Churn on GitHub](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

### Target Variable

The target variable for this project is the `Churn` column. This is a binary classification task.

- **`Churn`**: Indicates whether the customer has left the company within the last month.
  - **'Yes'**: The customer has churned.
  - **'No'**: The customer has not churned.
