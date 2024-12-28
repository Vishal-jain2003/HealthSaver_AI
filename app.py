import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import MultiLabelBinarizer

# Load the necessary data
sym_des = pd.read_csv('datasets/symtoms_df.csv')  # Assuming this has a list of symptoms
precautions = pd.read_csv('datasets/precautions_df.csv')
workout = pd.read_csv('datasets/workout_df.csv')
description = pd.read_csv('datasets/description.csv')
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv('datasets/diets.csv')

# Load the trained model (support vector classifier in this case)
svc = pickle.load(open("models/svc.pkl", 'rb'))


# Function to preprocess and predict
def preprocess_and_predict(symptoms):
    # Assuming sym_des contains all possible symptoms
    all_symptoms = sym_des['Symptom'].tolist()  # List of all symptoms in the training set

    # One-hot encoding for the symptoms entered by the user
    mlb = MultiLabelBinarizer()
    mlb.fit([all_symptoms])  # Fit the encoder with the list of all symptoms

    # Convert the input symptoms to a one-hot encoded vector
    input_vector = mlb.transform([symptoms.split(',')])[0]

    # Make the prediction
    prediction = svc.predict([input_vector])
    return prediction


# Streamlit UI
st.title("HealthSaver AI")
st.write("Enter symptoms for disease prediction")

# Input field for symptoms (comma separated)
symptoms_input = st.text_input("Type Symptoms (comma separated)", "")

if st.button("Get Recommendation"):
    if symptoms_input:
        # Make prediction based on user input
        prediction = preprocess_and_predict(symptoms_input)

        # Display the prediction result
        st.write(f"Predicted Disease: {prediction[0]}")

        # Fetch and display additional information about the predicted disease
        disease = prediction[0]

        # Display disease description
        disease_description = description[description['Disease'] == disease]['Description'].values
        if disease_description:
            st.write(f"Description: {disease_description[0]}")

        # Display disease precautions
        disease_precautions = precautions[precautions['Disease'] == disease]['Precaution'].values
        if disease_precautions:
            st.write(f"Precautions: {', '.join(disease_precautions)}")

        # Display recommended workouts for the disease
        disease_workouts = workout[workout['Disease'] == disease]['Workout'].values
        if disease_workouts:
            st.write(f"Recommended Workouts: {', '.join(disease_workouts)}")

        # Display medications for the disease
        disease_medications = medications[medications['Disease'] == disease]['Medication'].values
        if disease_medications:
            st.write(f"Recommended Medications: {', '.join(disease_medications)}")

        # Display diet for the disease
        disease_diet = diets[diets['Disease'] == disease]['Diet'].values
        if disease_diet:
            st.write(f"Recommended Diet: {', '.join(disease_diet)}")
    else:
        st.write("Please enter symptoms.")
