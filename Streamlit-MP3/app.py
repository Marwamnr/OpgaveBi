import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from PIL import Image
import joblib  # for saving/loading models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load logo image
logo = Image.open('media/logo.png')

st.set_page_config(
    page_title="Employee Analytics App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a Bug': 'https://github.com/your-repo/issues',
        'About': 'This is a cool app for employee analytics!'
    }
)

# Sidebar UI with option menu for tasks
st.sidebar.image(logo, width=200)
st.sidebar.header("Try Me!", divider='rainbow')

menu = option_menu(
    menu_title="Select Task",
    options=[
        "1. Data Wrangling & Exploration",
        "2. Income Prediction (Regression)",
        "3. Attrition Prediction (Classification)",
        "4. Employee Segmentation (Clustering)",
        "5. Model Application"
    ],
    icons=["file-earmark-spreadsheet", "graph-up", "person-check", "diagram-3", "play-circle"],
    menu_icon="cast",
    default_index=0
)

# Banner
banner = """
    <div style="background-color:#385c7f; padding:10px; margin-bottom:15px;">
        <h2 style="color:white; text-align:center;">MiniProject 3 - Employee Analytics</h2>
    </div>
"""
st.markdown(banner, unsafe_allow_html=True)

# Global variables for data/models
DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"  # example path, adjust as needed
MODEL_DIR = "models"

# Utility functions for loading data and models
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def save_model(model, filename):
    joblib.dump(model, os.path.join(MODEL_DIR, filename))

def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))

# Task 1: Data wrangling and exploration
if menu == "1. Data Wrangling & Exploration":
    st.header("Data Wrangling and Exploration")

    st.write("Loading dataset...")
    df = load_data()
    st.write(f"Data shape: {df.shape}")
    st.dataframe(df.head())

    st.subheader("Basic Data Info")
    st.write(df.info())

    st.subheader("Data Cleaning")
    st.write("Implement cleaning steps here (e.g., missing value handling)")

    st.subheader("Feature Selection")
    st.write("Select features most relevant for modeling")

    # Example: show correlation heatmap, distributions, etc.
    if st.checkbox("Show correlation matrix"):
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot(fig)

# Task 2: Supervised ML - Linear Regression for income prediction
elif menu == "2. Income Prediction (Regression)":
    st.header("Income Prediction using Linear Regression")

    df = load_data()

    # Feature selection (example)
    features = st.multiselect("Select features for model:", df.columns.tolist(), default=["Age", "YearsAtCompany"])
    target = "Income"

    if st.button("Train Model"):
        if len(features) == 0:
            st.error("Please select at least one feature")
        else:
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success("Model trained!")
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"R^2 Score: {r2:.2f}")

            # Save model
            save_model(model, "LinearRegression_model.pkl")
            st.info("Model saved as LinearRegression_model.pkl")

    st.subheader("Make Prediction")
    input_data = {}
    for f in features:
        val = st.number_input(f"Input {f}", value=float(df[f].median()))
        input_data[f] = val
    if st.button("Predict Income"):
        model = load_model("LinearRegression_model.pkl")
        input_df = pd.DataFrame([input_data])
        pred = model.predict(input_df)[0]
        st.write(f"Predicted Income: ${pred:,.2f}")

# Task 3: Supervised ML - Classification for attrition prediction
elif menu == "3. Attrition Prediction (Classification)":
    st.header("Attrition Prediction using Classification")

    df = load_data()

    target = "Attrition"
    features = st.multiselect("Select features for classification:", df.columns.tolist(), default=["Age", "JobSatisfaction"])

    if st.button("Train Classifier"):
        if len(features) == 0:
            st.error("Select at least one feature")
        else:
            X = df[features]
            y = df[target]

            # Encode target if needed
            if y.dtype == 'object':
                y = y.map({'Yes': 1, 'No': 0})

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(random_state=42)
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            st.success("Classifier trained!")
            st.write(f"Accuracy: {acc:.2f}")
            st.text(classification_report(y_test, y_pred))

            save_model(clf, "RandomForestClassifier_model.pkl")
            st.info("Model saved as RandomForestClassifier_model.pkl")

    st.subheader("Make Prediction")
    input_data = {}
    for f in features:
        val = st.number_input(f"Input {f}", value=float(df[f].median()))
        input_data[f] = val

    if st.button("Predict Attrition"):
        clf = load_model("RandomForestClassifier_model.pkl")
        input_df = pd.DataFrame([input_data])
        pred = clf.predict(input_df)[0]
        st.write("Attrition Prediction:", "Yes" if pred == 1 else "No")

# Task 4: Unsupervised ML - Clustering
elif menu == "4. Employee Segmentation (Clustering)":
    st.header("Employee Segmentation using Clustering")

    df = load_data()
    features = st.multiselect("Select features for clustering:", df.select_dtypes(include=np.number).columns.tolist(), default=["Age", "Income", "YearsAtCompany"])

    if st.button("Run Clustering"):
        if len(features) == 0:
            st.error("Select at least one feature")
        else:
            X = df[features]

            max_clusters = st.slider("Max number of clusters to test", 2, 10, 5)
            silhouette_scores = []
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X)
                score = silhouette_score(X, labels)
                silhouette_scores.append((k, score))

            best_k, best_score = max(silhouette_scores, key=lambda x: x[1])

            st.write("Silhouette scores by number of clusters:")
            for k, score in silhouette_scores:
                st.write(f"Clusters: {k}, Silhouette Score: {score:.3f}")

            st.success(f"Best number of clusters: {best_k} with silhouette score {best_score:.3f}")

            # Final clustering with best_k
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(X)
            st.dataframe(df.head())

            # Save model
            save_model(kmeans, "KMeans_model.pkl")
            st.info("Clustering model saved as KMeans_model.pkl")

# Task 5: Model Application UI
elif menu == "5. Model Application":
    st.header("Apply Saved Models")

    model_choice = st.selectbox("Select model to apply:", ["Income Prediction", "Attrition Prediction", "Employee Segmentation"])

    if model_choice == "Income Prediction":
        model = load_model("LinearRegression_model.pkl")
        st.write("Input features for income prediction:")
        # Example fixed features, ideally load dynamically or save feature list alongside model
        features = ["Age", "YearsAtCompany"]
        input_data = {}
        for f in features:
            val = st.number_input(f"Input {f}", value=40.0)
            input_data[f] = val
        if st.button("Predict Income"):
            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)[0]
            st.write(f"Predicted Income: ${pred:,.2f}")

    elif model_choice == "Attrition Prediction":
        model = load_model("RandomForestClassifier_model.pkl")
        st.write("Input features for attrition prediction:")
        features = ["Age", "JobSatisfaction"]
        input_data = {}
        for f in features:
            val = st.number_input(f"Input {f}", value=3.0)
            input_data[f] = val
        if st.button("Predict Attrition"):
            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)[0]
            st.write("Attrition Prediction:", "Yes" if pred == 1 else "No")

    elif model_choice == "Employee Segmentation":
        model = load_model("KMeans_model.pkl")
        st.write("Input features for clustering:")
        features = ["Age", "Income", "YearsAtCompany"]
        input_data = {}
        for f in features:
            val = st.number_input(f"Input {f}", value=40.0)
            input_data[f] = val
        if st.button("Assign Cluster"):
            input_df = pd.DataFrame([input_data])
            cluster = model.predict(input_df)[0]
            st.write(f"Employee assigned to cluster: {cluster}")

st.markdown(
    """
    ### To learn more
    - Check out [Streamlit Documentation](https://docs.streamlit.io)
    - Contact me by [email](mailto://tdi@cphbusiness.dk)
    """
)
