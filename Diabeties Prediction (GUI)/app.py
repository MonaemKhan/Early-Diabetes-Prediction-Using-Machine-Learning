import streamlit as st
import pickle
import numpy as np
import sklearn

st.title("Diabetes Prediction")

Pregnancies = st.text_input('Number Of times pregnant')

Glucose = st.text_input('Oral Glucose Tolerance Test (2 hour)')

BloodPressure = st.text_input('Diastolic Blood Pressure (mm Hg)')

SkinThickness = st.text_input('Triceps Skin Fold Thickness (mm)')

Insulin = st.text_input('2-Hour Serum Insulin (micro U/ml')

BMI = st.text_input('Body Mass Index (BMI)')

DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')

Age = st.text_input('Age in Years')

mod = pickle.load(open('model.pkl', 'rb'))
sac = pickle.load(open('scaler.pkl', 'rb'))

if st.button('Predict Result'):
    data = np.zeros(8)
    data[0] =int(Pregnancies)
    data[1] =int(Glucose)
    data[2] =int(BloodPressure)
    data[3] =int(SkinThickness)
    data[4] =int(Insulin)
    data[5] =float(BMI)
    data[6] =float(DiabetesPedigreeFunction)
    data[7] =int(Age)

    data_scal = sac.transform([data])
    result = mod.predict(data_scal)[0]
    if result==1:
        st.subheader('Oh No!! you have a high chance of diabetes')
    else:
        st.subheader('Hurrah!! you are safe')