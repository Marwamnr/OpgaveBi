import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Set Streamlit Page Config
st.set_page_config(page_title="HR Employee Clustering Analysis", layout="wide")

# Title
st.title("HR Employee Clustering Analysis")
st.write("Analyze employee segmentation using unsupervised learning.")

def main():
    # Load Data
    DATA_PATH = "./data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(DATA_PATH)
    st.write(f"### Dataset Shape: {df.shape}")
    st.dataframe(df.head())

    # Data Quality Check
    st.subheader("Data Quality Check")
    st.write("#### Missing Values per Column:")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])
    duplicates = df.duplicated().sum()
    st.write(f"#### Total Duplicate Rows: {duplicates}")

    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    # Select features for clustering
    selected_features = [
        'JobLevel', 'TotalWorkingYears', 'YearsAtCompany', 
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'NumCompaniesWorked',
        'PerformanceRating', 'JobInvolvement', 'TrainingTimesLastYear', 'PercentSalaryHike',
        'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance', 'RelationshipSatisfaction',
        'MonthlyIncome', 'StockOptionLevel', 'DistanceFromHome'
    ]
    df['OverTime_Numeric'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    df['BusinessTravel_Numeric'] = df['BusinessTravel'].map({'Non-Travel': 0, 'Travel_Rarely': 1, 'Travel_Frequently': 2})
    selected_features.extend(['OverTime_Numeric', 'BusinessTravel_Numeric'])

    # Data scaling
    st.subheader("Scaling Selected Features")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[selected_features])

    # Optimal cluster selection
    st.subheader("Optimal Cluster Selection")
    silhouette_scores = []
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        score = silhouette_score(features_scaled, labels)
        silhouette_scores.append(score)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), silhouette_scores, marker='o', linewidth=2, markersize=8)
    plt.title('Silhouette Score per Cluster Count')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    st.pyplot(plt)

    # Choose optimal number of clusters
    optimal_clusters = np.argmax(silhouette_scores) + 2
    st.write(f"Optimal number of clusters: {optimal_clusters}")

    # Final clustering
    final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    df['cluster'] = final_kmeans.fit_predict(features_scaled)

    # Cluster Analysis
    st.subheader("Cluster Analysis")
    st.write(df.groupby('cluster')[selected_features].mean().T)

    # PCA for visualization
    st.subheader("PCA Visualization")
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['cluster'] = df['cluster']

    # Plot PCA
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='viridis', s=100, alpha=0.7)
    plt.title('Employee Clusters (PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    st.pyplot(plt)


if __name__ == '__main__':
    main()
