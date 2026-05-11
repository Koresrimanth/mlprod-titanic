import streamlit as st
import requests


st.title(
    "Titanic Survival Prediction"
)

st.write(
    "Enter passenger details"
)


# Inputs
pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

age = st.slider(
    "Age",
    1,
    100,
    25
)

sibsp = st.number_input(
    "Siblings/Spouses",
    0,
    10,
    0
)

parch = st.number_input(
    "Parents/Children",
    0,
    10,
    0
)

fare = st.number_input(
    "Fare",
    0.0,
    500.0,
    50.0
)

embarked = st.selectbox(
    "Embarked",
    ["S", "C", "Q"]
)


# Predict button
if st.button("Predict"):

    input_data = {

        "Pclass": pclass,

        "Sex": sex,

        "Age": age,

        "SibSp": sibsp,

        "Parch": parch,

        "Fare": fare,

        "Embarked": embarked
    }

    # Call FastAPI
    response = requests.post(

        "http://127.0.0.1:8000/predict",

        json=input_data
    )

    result = response.json()

    # Display result
    st.subheader(
        f"Prediction: {result['prediction']}"
    )

    st.write(
        f"Survival Probability: {result['survival_probability']}"
    )