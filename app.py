
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

@st.cache_resource
def train_model():
    # Load your existing dataset
    data = pd.read_csv("diabetes.csv")

    # Separate input features and target
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # Standardize features and train the SVM model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel="linear")
    model.fit(X_scaled, y)

    return scaler, model

scaler, model = train_model()

st.title("🩺 Diabetes Risk Predictor")
st.write(
    "Enter the health measurements below to get a "
    "machine learning prediction."
)

st.info(
    "This is an educational prediction tool, not a medical diagnosis."
)

with st.form("prediction_form"):
    st.subheader("Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=30, value=1
        )
        glucose = st.number_input(
            "Glucose", min_value=0, max_value=300, value=120
        )
        blood_pressure = st.number_input(
            "Blood Pressure", min_value=0, max_value=200, value=70
        )
        skin_thickness = st.number_input(
            "Skin Thickness", min_value=0, max_value=150, value=20
        )

    with col2:
        insulin = st.number_input(
            "Insulin", min_value=0, max_value=1000, value=79
        )
        bmi = st.number_input(
            "BMI", min_value=0.0, max_value=100.0,
            value=25.0, step=0.1
        )
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0, max_value=3.0,
            value=0.5, step=0.01
        )
        age = st.number_input(
            "Age", min_value=1, max_value=120, value=30
        )

    submitted = st.form_submit_button(
        "Predict Diabetes Risk",
        type="primary",
        use_container_width=True
    )

if submitted:
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    # Apply the same scaler used during model training
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning(
            "The model predicts a higher likelihood of diabetes. "
            "Please consult a qualified healthcare professional."
        )
    else:
        st.success(
            "The model predicts a lower likelihood of diabetes. "
            "This does not rule out diabetes."
        )

    st.caption(
        "Predictions are based on the dataset and model. "
        "They should not be used to make medical decisions."
    )
