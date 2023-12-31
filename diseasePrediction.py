import pickle
import streamlit as st
from Ecg import  ECG
from streamlit_option_menu import option_menu

# loading the saved models
diabetes_model = pickle.load(open('saved models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('saved models/heart_disease_model.sav', 'rb'))

# st.header('Disease Prediction System')

# page title
st.title('Health Prediction using ML')
    
# getting the input data from the user
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
    Pregnancies = st.text_input('Number of Pregnancies')
        
with col2:
    Glucose = st.text_input('Glucose Level')
    
# with col3:
#     BloodPressure = st.text_input('Blood Pressure value')
    
with col1:
    SkinThickness = st.text_input('Skin Thickness value')
    
with col2:
    Insulin = st.text_input('Insulin Level')
    
with col3:
    BMI = st.text_input('BMI value')
    
with col3:
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
# with col2:
#     Age = st.text_input('Age of the Person')
    


# page title
# st.title('Basic data')

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
        
with col2:
    thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
    
# code for Prediction
# code for Prediction
diab_diagnosis = ''
heart_diagnosis = ''
prediction_probability = 0.0

# Convert input values to numeric
age = 0.0
sex = 0.0
cp = 0.0
trestbps = 0.0
chol = 0.0
fbs = 0.0
restecg = 0.0
thalach = 0.0
exang = 0.0
oldpeak = 0.0
slope = 0.0
ca = 0.0
thal = 0.0

    # Reshape the input data to a 2D array
input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]



#intialize ecg object
ecg = ECG()
#get the uploaded image
uploaded_file = st.file_uploader("Choose a file")


    # creating a button for Prediction
if st.button('Predict Health Result'):
    heart_prediction = heart_disease_model.predict(input_data)[0]
    diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, trestbps, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, age]])

    # Get the prediction probability
    prediction_probability = heart_disease_model.predict_proba(input_data)[0][1]

    # if heart_prediction == 1:
    #     heart_diagnosis = 'The person is having heart disease'
    # else:
    #     heart_diagnosis = 'The person does not have any heart disease'
    
    if prediction_probability <= 0.2:
        heart_diagnosis = 'Heart is in Good Condition'
    elif 0.2 < prediction_probability <= 0.4:
        heart_diagnosis = 'Stage 1 Heart Disease (work on your health)'
    elif 0.4 < prediction_probability <= 0.6:
        heart_diagnosis = 'Stage 2 Heart Disease (Precautionary Measures Required)'
    elif 0.6 < prediction_probability <= 0.8:
        heart_diagnosis = 'Stage 3 Heart Disease (Consult a doctor)'
    else:
        heart_diagnosis = 'Stage 4 Heart Disease (Critical Condition)'

    
        
    if diab_prediction[0] == 1:
        diab_diagnosis = 'The person is diabetic'
    else:
        diab_diagnosis = 'The person is not diabetic'


st.success(f'Heart Health Score: {prediction_probability:.2f}')
st.success(heart_diagnosis)
st.success(diab_diagnosis)


if uploaded_file is not None:
  """#### **UPLOADED IMAGE**"""
  # call the getimage method
  ecg_user_image_read = ecg.getImage(uploaded_file)
  #show the image
  st.image(ecg_user_image_read)

  """#### **GRAY SCALE IMAGE**"""
  #call the convert Grayscale image method
  ecg_user_gray_image_read = ecg.GrayImgae(ecg_user_image_read)
  
  #create Streamlit Expander for Gray Scale
  my_expander = st.expander(label='Gray SCALE IMAGE')
  with my_expander: 
    st.image(ecg_user_gray_image_read)
  
  """#### **DIVIDING LEADS**"""
   #call the Divide leads method
  dividing_leads=ecg.DividingLeads(ecg_user_image_read)

  #streamlit expander for dividing leads
  my_expander1 = st.expander(label='DIVIDING LEAD')
  with my_expander1:
    st.image('Leads_1-12_figure.png')
    st.image('Long_Lead_13_figure.png')
  
  """#### **PREPROCESSED LEADS**"""
  #call the preprocessed leads method
  ecg_preprocessed_leads = ecg.PreprocessingLeads(dividing_leads)

  #streamlit expander for preprocessed leads
  my_expander2 = st.expander(label='PREPROCESSED LEAD')
  with my_expander2:
    st.image('Preprossed_Leads_1-12_figure.png')
    st.image('Preprossed_Leads_13_figure.png')
  
  """#### **EXTRACTING SIGNALS(1-12)**"""
  #call the sognal extraction method
  ec_signal_extraction = ecg.SignalExtraction_Scaling(dividing_leads)
  my_expander3 = st.expander(label='CONOTUR LEADS')
  with my_expander3:
    st.image('Contour_Leads_1-12_figure.png')
  
  """#### **CONVERTING TO 1D SIGNAL**"""
  #call the combine and conver to 1D signal method
  ecg_1dsignal = ecg.CombineConvert1Dsignal()
  my_expander4 = st.expander(label='1D Signals')
  with my_expander4:
    st.write(ecg_1dsignal)
    
  """#### **PERFORM DIMENSINALITY REDUCTION**"""
  #call the dimensinality reduction funciton
  ecg_final = ecg.DimensionalReduciton(ecg_1dsignal)
  my_expander4 = st.expander(label='Dimensional Reduction')
  with my_expander4:
    st.write(ecg_final)
  
#   """#### **PASS TO PRETRAINED ML MODEL FOR PREDICTION**"""
# #   call the Pretrainsed ML model for prediction
#   ecg_model=ecg.ModelLoad_predict(ecg_final)
#   my_expander5 = st.expander(label='PREDICTION')
#   with my_expander5:
#     st.write(ecg_model)