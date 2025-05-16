import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set Page Configuration
st.set_page_config(page_title="Employee Data Analysis Dashboard", layout="wide")

# Title
st.title("Employee Data Analysis Dashboard")

# Load Data
DATA_PATH = "./data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
data = pd.read_csv(DATA_PATH)
st.write(f"### Dataset Shape: {data.shape}")
st.dataframe(data.head())

# Data Quality Check
st.subheader("Data Quality Check")
missing_values = data.isnull().sum()
duplicates = data.duplicated().sum()
st.write("#### Missing Values per Column:")
st.write(missing_values[missing_values > 0])
st.write(f"#### Total Duplicate Rows: {duplicates}")

# Descriptive Statistics
st.subheader("Descriptive Statistics")
st.write(data.describe())

# Visualize Distribution of Numeric Features
st.subheader("Numerical Data Distribution")
numerical_data = data.select_dtypes(include=['float64', 'int64'])
skew_values = numerical_data.apply(skew)

for col in numerical_data.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(numerical_data[col], kde=True, ax=ax)
    ax.set_title(f"{col} - Skew: {skew_values[col]:.2f}")
    st.pyplot(fig)

# Log Transformation for Highly Skewed Columns
st.subheader("Log Transformation for High Skewness")
skewed_cols = [col for col in numerical_data.columns if abs(skew_values[col]) > 1]

for col in skewed_cols:
    data[col] = np.log1p(data[col])

skew_values_after = data[skewed_cols].apply(skew)
st.write("### Skewness After Transformation")
st.write(skew_values_after)

# Correlation Matrix
st.subheader("Correlation Matrix")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)

# Data Scaling
st.subheader("Scaling Selected Numerical Features")
scaler = StandardScaler()
selected_features = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'YearsAtCompany']
data[selected_features] = scaler.fit_transform(data[selected_features])
st.write(data[selected_features].head())

# Train-Test Split
st.subheader("Train-Test Split")
X = data.drop(columns=['Attrition'])
y = pd.get_dummies(data['Attrition'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
st.write(f"Training Data Shape: {X_train.shape}")
st.write(f"Testing Data Shape: {X_test.shape}")

# Linear Regression Model
st.subheader("Linear Regression Model Training")
model = LinearRegression()
model.fit(X_train, y_train)

# Validation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
st.write(f"Train R² Score: {train_r2:.2f}")
st.write(f"Test R² Score: {test_r2:.2f}")
