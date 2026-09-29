import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient information below to get a machine-learning prediction.")

@st.cache_resource
def train_model():
    data = pd.read_csv("dataset_heart.csv")
    data.columns = data.columns.str.strip()
    X = data.drop("heart disease", axis=1)
    y = data["heart disease"].map({1: 0, 2: 1})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    return model, scaler, X.columns.tolist()

model, scaler, feature_names = train_model()
data = pd.read_csv("dataset_heart.csv")

st.subheader("Patient Information")
inputs = {}

for col in feature_names:
    values = pd.to_numeric(data[col], errors="coerce").dropna()
    if (values % 1 == 0).all() and values.nunique() <= 10:
        inputs[col] = st.selectbox(col, sorted(values.astype(int).unique()))
    elif (values % 1 == 0).all():
        inputs[col] = st.number_input(
            col, min_value=int(values.min()), max_value=int(values.max()),
            value=int(values.median())
        )
    else:
        inputs[col] = st.number_input(
            col, min_value=float(values.min()), max_value=float(values.max()),
            value=float(values.median())
        )

if st.button("Predict", type="primary"):
    row = pd.DataFrame([inputs], columns=feature_names)
    prediction = int(model.predict(scaler.transform(row))[0])

    if prediction == 1:
        st.error("Prediction: Heart Disease Class")
    else:
        st.success("Prediction: No Heart Disease Class")

    st.caption("Educational machine-learning prediction only; not a medical diagnosis.")
