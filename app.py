# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, 'data', 'Titanic-Dataset.csv')
# fallback if user didn't rename uploaded file
if not os.path.exists(DATA_PATH):
    alt = '/mnt/data/Titanic-Dataset.csv'
    if os.path.exists(alt):
        DATA_PATH = alt

MODEL_PATH = os.path.join(ROOT, 'model.pkl')

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)

    # Convert object columns to strings
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)

    # Ensure numeric columns are numeric
    numeric_cols = ['Pclass','Age','SibSp','Parch','Fare','Survived']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df



@st.cache_data
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

def preprocess_sample_input(df_sample):
    # called only to display examples or for prediction via pipeline,
    # model pipeline contains preprocessing so no manual transform needed
    return df_sample

st.title("Titanic Survival Prediction")
st.markdown("A demo Streamlit app that loads a trained model pipeline and predicts passenger survival.")

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Home", "Data Explorer", "Visualizations", "Model Prediction", "Model Performance", "About"])

# Load data and model
if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found. Put your CSV into `data/dataset.csv` (or ensure {DATA_PATH}).")
    st.stop()

df = load_data(DATA_PATH)

# Convert all object columns to strings for Streamlit / PyArrow compatibility
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)


model = load_model(MODEL_PATH)

if page == "Home":
    st.header("Project overview")
    st.markdown("""
    - Dataset: Titanic (classification: survived / not survived)
    - Model: pipeline with preprocessing + classifier (saved as `model.pkl`)
    - Use the **Model Prediction** page to try predictions interactively.
    """)

elif page == "Data Explorer":
    st.header("Data overview")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.dataframe(df.sample(10, random_state=42))

    st.subheader("Filtering")
    col1, col2, col3 = st.columns(3)
    with col1:
        sex = st.multiselect("Sex", options=df['Sex'].dropna().unique().tolist(), default=df['Sex'].dropna().unique().tolist())
    with col2:
        pclass = st.multiselect("Pclass", options=sorted(df['Pclass'].dropna().unique().tolist()), default=sorted(df['Pclass'].dropna().unique().tolist()))
    with col3:
        embarked = st.multiselect("Embarked", options=df['Embarked'].dropna().unique().tolist(), default=df['Embarked'].dropna().unique().tolist())

    min_age = float(df['Age'].min())
    max_age = float(df['Age'].max())
    age_range = st.slider("Age range", min_value=float(min_age), max_value=float(max_age), value=(min_age, max_age))
    fare_min = float(df['Fare'].min())
    fare_max = float(df['Fare'].max())
    fare_range = st.slider("Fare range", min_value=float(fare_min), max_value=float(fare_max), value=(fare_min, fare_max))

    filtered = df[
        (df['Sex'].isin(sex)) &
        (df['Pclass'].isin(pclass)) &
        (df['Embarked'].isin(embarked)) &
        (df['Age'].between(age_range[0], age_range[1], inclusive="both")) &
        (df['Fare'].between(fare_range[0], fare_range[1], inclusive="both"))
    ].copy()  # <-- add .copy() here

    # Ensure Arrow compatibility for filtered dataframe
    for col in filtered.select_dtypes(include=['object']).columns:
        filtered.loc[:, col] = filtered[col].astype(str)  # <-- use .loc for assignment

    st.write("Filtered rows:", filtered.shape[0])
    st.dataframe(filtered.head(20))

elif page == "Visualizations":
    st.header("Interactive Visualizations")
    viz_choice = st.selectbox("Choose plot", ["Age histogram", "Survival by Pclass", "Fare vs Age (scatter)"])
    if viz_choice == "Age histogram":
        fig = px.histogram(df, x='Age', nbins=30, title="Age distribution (non-null ages)", hover_data=['Survived'])
        st.plotly_chart(fig, use_container_width=True)
    elif viz_choice == "Survival by Pclass":
        fig = px.bar(df.groupby(['Pclass','Survived'])['PassengerId'].count().reset_index(name='count'),
                     x='Pclass', y='count', color='Survived', barmode='group',
                     title="Survival by Pclass")
        st.plotly_chart(fig, use_container_width=True)
    elif viz_choice == "Fare vs Age (scatter)":
        fig = px.scatter(df, x='Age', y='Fare', color='Survived', hover_data=['Pclass','Sex'])
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Prediction":
    st.header("Make a prediction")
    if model is None:
        st.warning("Model not found. Train the model first (run the notebook or training script). Expected file: model.pkl")
    else:
        st.markdown("Enter passenger details and click **Predict**")
        col1, col2, col3 = st.columns(3)
        with col1:
            pclass = st.selectbox("Pclass", [1,2,3], index=2)
            sex = st.selectbox("Sex", ["male","female"])
            age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0)
        with col2:
            sibsp = st.number_input("SibSp (siblings/spouses aboard)", min_value=0, max_value=10, value=0)
            parch = st.number_input("Parch (parents/children aboard)", min_value=0, max_value=10, value=0)
            fare = st.number_input("Fare", min_value=0.0, max_value=1000.0, value=32.0)
        with col3:
            embarked = st.selectbox("Embarked", sorted(df['Embarked'].dropna().unique().tolist()))

        sample = pd.DataFrame([{
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked
        }])

        # FIX: enforce correct types
        sample['Sex'] = sample['Sex'].astype(str)
        sample['Embarked'] = sample['Embarked'].astype(str)
        for col in ['Pclass','Age','SibSp','Parch','Fare']:
            sample[col] = pd.to_numeric(sample[col], errors='coerce')

        # Prediction
        try:
            proba = model.predict_proba(sample)[0]
            pred = model.predict(sample)[0]
            st.success(f"Predicted: **{'Survived' if pred==1 else 'Not survived'}**")
            st.info(f"Probability survived: {proba[1]:.3f}  —  Not survived: {proba[0]:.3f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

        st.markdown("Example model input (for reference):")
        # Ensure Arrow compatibility for transposed sample
        sample_T = sample.T.copy()
        sample_T[0] = sample_T[0].astype(str)
        st.table(sample_T)

elif page == "Model Performance":
    st.header("Model evaluation (on test split)")
    if model is None:
        st.warning("Model not found. Please train with the notebook or training script first.")
    else:
        # Prepare evaluation dataset
        eval_df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']].dropna()
        X_eval = eval_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
        y_eval = eval_df['Survived']
        # small test-split for demonstration
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_eval, y_eval, test_size=0.2, random_state=42, stratify=y_eval)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy on local test split: **{acc:.3f}**")
        st.subheader("Classification report")
        cr = classification_report(y_test, y_pred, output_dict=True)
        st.json(cr)
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actual 0','Actual 1'], columns=['Pred 0','Pred 1'])
        st.table(cm_df)

elif page == "About":
    st.header("About")
    st.markdown("""
    This app uses a scikit-learn pipeline (preprocessing + classifier). 
    Use the notebook or `notebooks/training_script.py` to train and create `model.pkl`.
    """)
