import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE


def main():
    # Load Data
        DATA_PATH = "./data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
        data = pd.read_csv(DATA_PATH)
        st.write(f"### Dataset Shape: {data.shape}")
        st.dataframe(data.head())

         # Data Cleaning and Preprocessing
        st.subheader("Data Cleaning")
        st.write("Checking for missing values and duplicates...")
        st.write(data.isna().sum())
        st.write(f"Duplicates: {data.duplicated().sum()}")

        # Encoding
        st.subheader("Encoding Categorical Variables")
        data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})
        categorical_cols = data.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_cols:
            data[col] = le.fit_transform(data[col])

        # Data Scaling
        numerical_features = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'YearsAtCompany']
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        # Train-Test Split
        X = data.drop('Attrition', axis=1)
        y = data['Attrition']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # SMOTE for Class Imbalance
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Model Training
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train_resampled, y_train_resampled)
        y_pred = clf.predict(X_test)

        # Evaluation
        st.subheader("Model Evaluation")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(plt)


if __name__ == '__main__':
    main()
