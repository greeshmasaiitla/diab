import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('diab.csv')

# Preprocessing the data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
logistic_model = LogisticRegression(max_iter=500)
logistic_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=300, max_depth=4, max_features=3)
rf_model.fit(X_train, y_train)

svm_model = SVC(probability=True)
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf']}
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_svm_model = grid_search.best_estimator_

# Streamlit app layout
st.title("Diabetes Prediction using Machine Learning")
st.write("Please enter your health parameters below:")

# Sidebar with parameter reference guide
with st.sidebar:
    st.write("### Parameter Reference Guide")
    st.write("- **Glucose (mg/dL):**")
    st.write("  - Normal: 70–99")
    st.write("  - Pre-diabetes: 100–125")
    st.write("  - Diabetes: ≥126")
    st.write("- **Blood Pressure (mm Hg):**")
    st.write("  - Normal: <120/80")
    st.write("  - Hypertension: ≥140/90")
    st.write("- **BMI:**")
    st.write("  - Normal: 18.5–24.9")
    st.write("  - Overweight: 25–29.9")
    st.write("  - Obese: ≥30")
    st.write("- **Insulin (mu U/ml):**")
    st.write("  - Normal: 2–25")

# Input fields for user data
gender = st.selectbox("Select Your Gender", options=["Male", "Female"])
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=200)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100)
insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, max_value=500.0)
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=50.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5)
age = st.number_input("Age (years)", min_value=1, max_value=120)

# Button for prediction
if st.button("🔍 Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree_function, age]])
    
    input_data_scaled = scaler.transform(input_data)

    # Make predictions using all three models
    logistic_prediction = logistic_model.predict(input_data_scaled)[0]
    rf_prediction = rf_model.predict(input_data_scaled)[0]
    svm_prediction = best_svm_model.predict(input_data_scaled)[0]

    # Majority voting for final prediction using NumPy's unique function
    predictions = np.array([logistic_prediction, rf_prediction, svm_prediction])
    final_prediction = np.argmax(np.bincount(predictions))  # Get the most common prediction

    # Display final result without showing individual model predictions
    if final_prediction == 1:
        st.success("Final Prediction: You are likely to have diabetes.")
    else:
        st.success("Final Prediction: You are not likely to have diabetes.")
