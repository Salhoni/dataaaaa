import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np
from scipy import stats
import io
import base64

# Set page configuration at the very top
st.set_page_config(
    page_title="Interactive Data Analysis Web App",
    layout="wide",
    initial_sidebar_state="expanded",
    theme={"primaryColor": "#1f77b4"}
)

# Function to load the dataset
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to handle missing values
def handle_missing_values(data, method):
    if method == 'Drop rows':
        return data.dropna()
    elif method == 'Fill with mean':
        return data.fillna(data.mean())
    elif method == 'Fill with median':
        return data.fillna(data.median())

# Function to perform descriptive statistics
def descriptive_stats(data):
    st.write(data.describe())

# Function to create visualizations
def create_visualizations(data, x_col, y_col, plot_type):
    if plot_type == 'Bar Chart':
        fig = px.bar(data, x=x_col, y=y_col)
    elif plot_type == 'Line Plot':
        fig = px.line(data, x=x_col, y=y_col)
    elif plot_type == 'Histogram':
        fig = px.histogram(data, x=x_col)
    elif plot_type == 'Scatter Plot':
        fig = px.scatter(data, x=x_col, y=y_col)
    elif plot_type == 'Box Plot':
        fig = px.box(data, x=x_col, y=y_col)
    elif plot_type == '3D Scatter Plot':
        z_col = st.selectbox("Select Z-axis column", options=data.columns)
        fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col)
    st.plotly_chart(fig)

# Function to find correlations
def find_correlations(data):
    corr = data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Function to make predictions using different econometric models
def make_predictions(data, target_col, model_type):
    if target_col in data.columns:
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        if model_type == 'Linear Regression':
            model = LinearRegression()
        elif model_type == 'Multiple Regression':
            model = LinearRegression()
        elif model_type == 'Logistic Regression':
            model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")
        st.write(f"R^2 Score: {r2_score(y_test, predictions)}")
        if model_type == 'Logistic Regression':
            accuracy = accuracy_score(y_test, predictions.round())
            st.write(f"Accuracy: {accuracy}")

# Function to perform advanced statistical tests
def advanced_stats(data):
    st.write("### Advanced Statistical Analysis")
    # t-test
    st.write("#### t-test")
    group_col = st.selectbox("Select column for grouping (t-test)", options=data.columns)
    value_col = st.selectbox("Select numeric column for t-test", options=data.select_dtypes(include='number').columns)
    groups = data[group_col].unique()
    if len(groups) == 2:
        group1 = data[data[group_col] == groups[0]][value_col]
        group2 = data[data[group_col] == groups[1]][value_col]
        t_stat, p_val = stats.ttest_ind(group1, group2)
        st.write(f"t-statistic: {t_stat}")
        st.write(f"p-value: {p_val}")
    else:
        st.write("t-test requires exactly two groups.")
    # chi-square test
    st.write("#### Chi-Square Test")
    cat_col1 = st.selectbox("Select first categorical column for chi-square", options=data.select_dtypes(include='object').columns)
    cat_col2 = st.selectbox("Select second categorical column for chi-square", options=data.select_dtypes(include='object').columns)
    contingency_table = pd.crosstab(data[cat_col1], data[cat_col2])
    chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
    st.write(f"Chi2: {chi2}")
    st.write(f"p-value: {p}")
    st.write(f"Degrees of freedom: {dof}")

# Function to download processed data
def download_data(data):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="processed_data.csv">Download Processed Data</a>'
    st.markdown(href, unsafe_allow_html=True)

# Function to generate a PDF report (placeholder)
def generate_report(data):
    st.write("### PDF Report")
    st.write("Report generation feature coming soon!")

# Main app
st.title("Interactive Data Analysis Web App")

# User authentication (placeholder)
st.write("## User Authentication")
st.write("Login system coming soon!")

# User input for data
input_type = st.radio("Choose input type", ["File Upload", "Text/Number Input"])
if input_type == "File Upload":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
else:
    user_input = st.text_area("Enter your data (comma-separated values)", "1,2,3\n4,5,6")
    data = pd.read_csv(io.StringIO(user_input), header=None)

if 'data' in locals():
    st.write("### Input Data")
    st.write(data)

    # Sidebar options
    st.sidebar.header("Filter Options")
    x_col = st.sidebar.selectbox("Select X-axis column", options=data.columns)
    y_col = st.sidebar.selectbox("Select Y-axis column", options=data.columns)
    plot_type = st.sidebar.radio("Select plot type", options=['Bar Chart', 'Line Plot', 'Histogram', 'Scatter Plot', 'Box Plot', '3D Scatter Plot'])

    if st.sidebar.checkbox("Apply Numeric Filter"):
        num_col = st.sidebar.selectbox("Select Numeric Column", options=data.select_dtypes(include='number').columns)
        min_val = st.sidebar.number_input("Minimum Value", value=float(data[num_col].min()))
        max_val = st.sidebar.number_input("Maximum Value", value=float(data[num_col].max()))
        data = data[(data[num_col] >= min_val) & (data[num_col] <= max_val)]

    if st.sidebar.checkbox("Apply Category Filter"):
        cat_col = st.sidebar.selectbox("Select Category Column", options=data.select_dtypes(include='object').columns)
        categories = st.sidebar.multiselect("Select Categories", options=data[cat_col].unique())
        data = data[data[cat_col].isin(categories)]

    # Descriptive statistics
    descriptive_stats(data)

    # Visualizations
    create_visualizations(data, x_col, y_col, plot_type)

    # Correlations
    st.write("### Correlations
