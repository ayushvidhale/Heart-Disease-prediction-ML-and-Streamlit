import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved models
diabetes_model = pickle.load(open('saved models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('saved models/heart_disease_model.sav', 'rb'))

# st.header('Disease Prediction System')

# page title
st.title('Diabetes Prediction using ML')
    
# getting the input data from the user
col1, col2, col3 = st.columns(3)
    
with col1:
    Pregnancies = st.text_input('Number of Pregnancies')
        
with col2:
    Glucose = st.text_input('Glucose Level')
    
with col3:
    BloodPressure = st.text_input('Blood Pressure value')
    
with col1:
    SkinThickness = st.text_input('Skin Thickness value')
    
with col2:
    Insulin = st.text_input('Insulin Level')
    
with col3:
    BMI = st.text_input('BMI value')
    
with col1:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
with col2:
    Age = st.text_input('Age of the Person')
    
# code for Prediction
diab_diagnosis = ''
    
# creating a button for Prediction
if st.button('Diabetes Test Result'):
    diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
    if diab_prediction[0] == 1:
        diab_diagnosis = 'The person is diabetic'
    else:
        diab_diagnosis = 'The person is not diabetic'
        
st.success(diab_diagnosis)

# page title
st.title('Heart Disease Prediction using ML')
    
col1, col2, col3 = st.columns(3)
    
with col1:
    age = st.text_input('Age')
        
with col2:
    sex = st.text_input('Sex')
        
with col3:
    cp = st.text_input('Chest Pain types')
        
with col1:
    trestbps = st.text_input('Resting Blood Pressure')
        
with col2:
    chol = st.text_input('Serum Cholestoral in mg/dl')
        
with col3:
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
with col1:
    restecg = st.text_input('Resting Electrocardiographic results')
        
with col2:
    thalach = st.text_input('Maximum Heart Rate achieved')
        
with col3:
    exang = st.text_input('Exercise Induced Angina')
        
with col1:
    oldpeak = st.text_input('ST depression induced by exercise')
        
with col2:
    slope = st.text_input('Slope of the peak exercise ST segment')
        
with col3:
    ca = st.text_input('Major vessels colored by flourosopy')
        
with col1:
    thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
     
# code for Prediction
heart_diagnosis = ''
prediction_probability = 0.0

# Convert input values to numeric
age = float(age)
sex = float(sex)
cp = float(cp)
trestbps = float(trestbps)
chol = float(chol)
fbs = float(fbs)
restecg = float(restecg)
thalach = float(thalach)
exang = float(exang)
oldpeak = float(oldpeak)
slope = float(slope)
ca = float(ca)
thal = float(thal)

    # Reshape the input data to a 2D array
input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

    # creating a button for Prediction
if st.button('Heart Disease Test Result'):
    heart_prediction = heart_disease_model.predict(input_data)[0]

    # Get the prediction probability
    prediction_probability = heart_disease_model.predict_proba(input_data)[0][1]

    if heart_prediction == 1:
        heart_diagnosis = 'The person is having heart disease'
    else:
        heart_diagnosis = 'The person does not have any heart disease'

st.success(f'Heart Health Score: {prediction_probability:.2f}')
st.success(heart_diagnosis)
    