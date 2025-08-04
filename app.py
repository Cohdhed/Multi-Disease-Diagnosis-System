import os
import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu


    
# Load model and preprocessing artifacts
script_directory = os.path.dirname(os.path.abspath(__file__))
#file_path_1 = os.path.join(script_directory, 'cancer_detection.joblib')
file_path_1 = os.path.join(script_directory, 'diabetes.joblib')
file_path_2 = os.path.join(script_directory, 'heart_disease.joblib')
file_path_3 = os.path.join(script_directory, 'liver_disease.joblib')
file_path_4 = os.path.join(script_directory, 'kidney_disease.joblib')

# Load the file using the absolute path
artifact_1 = joblib.load(file_path_1)
artifact_2 = joblib.load(file_path_2)
artifact_3 = joblib.load(file_path_3)
artifact_4 = joblib.load(file_path_4)



# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")


# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Liver Disease Prediction',
                            'Kidney Disease Prediction'],
                            
                           menu_icon='hospital-fill',
                           icons=['ribbon', 'ribbon', 'ribbon', 'ribbon'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies')

    with col2:
        Glucose = st.number_input('Glucose Level')

    with col3:
        BloodPressure = st.number_input('Blood Pressure value')

    with col1:
        SkinThickness = st.number_input('Skin Thickness value')

    with col2:
        Insulin = st.number_input('Insulin Level')

    with col3:
        BMI = st.number_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.number_input('Age of the Person')


    user_input = {
    'Pregnancies': Pregnancies,
    'Glucose': Glucose,
    'BloodPressure': BloodPressure,
    'SkinThickness': SkinThickness,
    'Insulin': Insulin,
    'BMI': BMI,
    'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
    'Age': Age,
    }

    # code for Prediction
    diab_diagnosis = ''

    

    def check_missing_inputs(data):
        return any(value == '' for value in data.values())


    def make_prediction(data):
        X = pd.DataFrame([data])
        
        prediction = artifact_1['model'].predict(X)[0]
        decision = 'The person is diabetic' if prediction == 1 else 'The person is not diabetic'
        return decision


    # Button to make prediction
    if st.button('Diabetes Test Result'):
        if check_missing_inputs(user_input):
            st.error("Please fill in all the fields.")
        else:
            result = make_prediction(user_input)
            st.success(result)



    

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age')

    with col2:
        sex = st.selectbox("Sex", ['','Male', 'Female'])

    with col3:
        cp = st.selectbox("Chest Pain type", ['','Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'])

    with col1:
        trestbps = st.number_input('Resting Blood Pressure')

    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ['','True', 'False'])

    with col1:
        restecg = st.selectbox("Resting Electrocardiographic results", 
                               ['', 'Normal', 
                                'Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)', 
                                'showing probable or definite left ventricular hypertrophy by Estes criteria'])

    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.selectbox('Exercise Induced Angina', ['', 'Yes', 'No'])

    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')

    with col2:
        slope = st.selectbox('Slope of the peak exercise ST segment', ['', 'Upsloping', 'Flat', 'Downsloping'])

    with col3:
        ca = st.selectbox('Major vessels colored by flourosopy', ['', 0,1,2,3,4])

    with col1:
        thal = st.selectbox('thal', ['', 'Fixed defect', 'Normal', 'Reversable defect'])




    # Mapping dictionaries for selectbox inputs
    sex_mapping = {
        'Male': 1,
        'Female': 0
    }

    fbs_mapping = { 
        'True': 1,
        'False': 0
    }

    restecg_mapping = {
        'Normal': 0,
        'Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)': 1,
        'showing probable or definite left ventricular hypertrophy by Estes criteria': 2
    }

    exang_mapping = {
        'Yes': 1,
        'No': 0
    }

    slope_mapping = {
        'Upsloping': 0,
        'Flat': 1,
        'Downsloping': 2
    }

    thal_mapping = {
        'Normal': 2,
        'Fixed defect': 1,
        'Reversable defect': 3
    }
    cp_mapping = {
        'Typical angina': 0,
        'Atypical angina': 1,
        'Non-anginal pain': 2,
        'Asymptomatic': 3
    }

    # Convert selected values to corresponding numerical values
    sex_value = sex_mapping.get(sex, '')
    cp_value = cp_mapping.get(cp, '')
    fbs_value = fbs_mapping.get(fbs, '')
    restecg_value = restecg_mapping.get(restecg, '')
    exang_value = exang_mapping.get(exang, '')
    slope_value = slope_mapping.get(slope, '')
    thal_value = thal_mapping.get(thal, '')



    user_input = {
    'age': age,
    'sex': sex_value,
    'cp': cp_value,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs_value,
    'restecg': restecg_value,
    'thalach': thalach,
    'exang': exang_value,
    'oldpeak': oldpeak,
    'slope': slope_value,
    'ca': ca,
    'thal': thal_value,
    }

    # code for Prediction
    heart_diagnosis = ''


    def check_missing_inputs(data):
        return any(value == '' for value in data.values())


    def make_prediction(data):
        df = pd.DataFrame([data])
        X = pd.DataFrame(artifact_2['preprocessing'].transform(df),
                     columns=artifact_2['preprocessing'].get_feature_names_out())
        
        prediction = artifact_2['model'].predict(X)[0]
        decision = 'The person is having heart disease' if prediction == 1 else 'The person does not have any heart disease'
        return decision


    # Button to make prediction
    if st.button('Heart Disease Test Result'):
        if check_missing_inputs(user_input):
            st.error("Please fill in all the fields.")
        else:
            result = make_prediction(user_input)
            st.success(result)





# Liver Disease Prediction Page
if selected == 'Liver Disease Prediction':

    # page title
    st.title('Liver Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=150, step=1)

    with col2:
        gender = st.selectbox("Gender", ['','Male', 'Female'])

    with col3:
        total_bilirubin = st.number_input('Total Bilirubin', min_value=0.0, max_value=100.0, step=0.1)

    with col1:
        direct_bilirubin = st.number_input('Direct Bilirubin', min_value=0.0, max_value=50.0, step=0.1)

    with col2:
        alkaline_phosphotase = st.number_input('Alkaline Phosphotase', min_value=0.0, max_value=5000.0, step=1.0)

    with col3:
        alamine_aminotransferase = st.number_input('Alamine Aminotransferase', min_value=0.0, max_value=5000.0, step=1.0)

    with col1:
        aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase', min_value=0.0, max_value=7000.0, step=1.0)

    with col2:
        total_protiens = st.number_input('Total Protiens', min_value=0.0, max_value=50.0, step=0.1)

    with col3:
        albumin = st.number_input('Albumin', min_value=0.0, max_value=50.0, step=0.1)

    with col1:
        albumin_and_globulin_ratio = st.number_input('Albumin and Globulin Ratio', min_value=0.0, max_value=50.0, step=0.1)

 
    gender_mapping = {
        'Male': 1,
        'Female': 0
    }

    # Convert selected values to corresponding numerical values
    gender_value = gender_mapping.get(gender, '')

    user_input = {
        'Age': age,
        'Gender': gender_value,
        'Total_Bilirubin': total_bilirubin,
        'Direct_Bilirubin': direct_bilirubin,
        'Alkaline_Phosphotase': alkaline_phosphotase,
        'Alamine_Aminotransferase': alamine_aminotransferase,
        'Aspartate_Aminotransferase': aspartate_aminotransferase,
        'Total_Protiens': total_protiens,
        'Albumin': albumin,
        'Albumin_and_Globulin_Ratio': albumin_and_globulin_ratio
    }


    # code for Prediction
    liver_disease_diagnosis = ''


    def check_missing_inputs(data):
        return any(value == '' for value in data.values())


    def make_prediction(data):
        df = pd.DataFrame([data])
        X = pd.DataFrame(artifact_3['preprocessing'].transform(df),
                        columns=artifact_3['preprocessing'].get_feature_names_out())
        
        prediction = artifact_3['model'].predict(X)[0]
        decision = 'Liver Disease Detected' if prediction == 1 else 'No Liver Disease Detected'
        return decision


    # Button to make prediction
    if st.button('Liver Disease Test Result'):
        if check_missing_inputs(user_input):
            st.error("Please fill in all the fields.")
        else:
            result = make_prediction(user_input)
            st.success(result)





# Kidney Disease Prediction Page
if selected == 'Kidney Disease Prediction':

    # page title
    st.title('Kidney Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, max_value=105, step=1)

    with col2:
        blood_pressure = st.number_input('Blood Pressure', min_value=0.0, max_value=500.0)

    with col3:
        specific_gravity = st.number_input('Specific Gravity', min_value=0.0, max_value=10.0)

    with col1:
        albumin = st.number_input('Albumin', min_value=0.0, max_value=10.0)

    with col2:
        sugar = st.number_input('Sugar', min_value=0.0, max_value=10.0)

    with col3:
        red_blood_cells = st.selectbox("Red Blood Cells", ['','Normal', 'Abnormal'])

    with col1:
        pus_cell = st.selectbox("Pus Cell", ['','Normal', 'Abnormal'])

    with col2:
        pus_cell_clumps = st.selectbox("Pus Cell Clumps", ['','Present', 'Not present'])

    with col3:
        bacteria = st.selectbox("Bacteria", ['','Present', 'Not present'])

    with col1:
        blood_glucose_random = st.number_input('Blood Glucose Random', min_value=0.0, max_value=1000.0)

    with col2:
        blood_urea = st.number_input('Blood Urea', min_value=0.0, max_value=1000.0)

    with col3:
        serum_creatinine = st.number_input('Serum Creatinine', min_value=0.0, max_value=200.0)

    with col1:
        sodium = st.number_input('Sodium', min_value=0.0, max_value=500.0) #HERE
    
    with col2:
        potassium = st.number_input('Potassium', min_value=0.0, max_value=100.0)

    with col3:
        haemoglobin = st.number_input('Haemoglobin', min_value=0.0, max_value=100.0)

    with col1:
        packed_cell_volume = st.number_input('Packed Cell Volume', min_value=0.0, max_value=200.0) 
    
    with col2:
        white_blood_cell_count = st.number_input('white Blood Cell Count', min_value=0.0, max_value=200000.0)

    with col3:
        red_blood_cell_count = st.number_input('Red Blood Cell Count', min_value=0.0, max_value=100.0)

    with col1:
        hypertension = st.selectbox("Hypertension", ['','Yes', 'No'])
    
    with col2:
        diabetes_mellitus = st.selectbox("Diabetes Mellitus", ['','Yes', 'No'])

    with col3:
        coronary_artery_disease = st.selectbox("Coronary Artery Disease", ['','Yes', 'No'])

    with col1:
        appetite = st.selectbox("Appetite", ['','Good', 'Poor'])
    
    with col2:
        peda_edema = st.selectbox("Pedal Edema", ['','Yes', 'No'])

    with col3:
        aanemia = st.selectbox("Aanemia", ['','Yes', 'No'])




    # Mapping dictionaries for selectbox inputs
    red_blood_cells_mapping = {
        'Normal': 1,
        'Abnormal': 0
    }

    pus_cell_mapping = { 
        'Normal': 1,
        'Abnormal': 0
    }

    pus_cell_clumps_mapping = {
        'Present': 1,
        'Not present': 0
    }

    bacteria_mapping = {
        'Present': 1,
        'Not present': 0
    }

    hypertension_mapping = {
        'Yes': 1,
        'No': 0
    }

    diabetes_mellitus_mapping = {
        'Yes': 1,
        'No': 0
    }
    
    coronary_artery_disease_mapping = {
        'Yes': 1,
        'No': 0
    }

    appetite_mapping = {
        'Good': 1,
        'Poor': 0
    }

    peda_edema_mapping = {
        'Yes': 1,
        'No': 0
    }

    aanemia_mapping = {
        'Yes': 1,
        'No': 0
    }

    # Convert selected values to corresponding numerical values
    red_blood_cells_value = red_blood_cells_mapping.get(red_blood_cells, '')
    pus_cell_value = pus_cell_mapping.get(pus_cell, '')
    pus_cell_clumps_value = pus_cell_clumps_mapping.get(pus_cell_clumps, '')
    bacteria_value = bacteria_mapping.get(bacteria, '')
    hypertension_value = hypertension_mapping.get(hypertension, '')
    diabetes_mellitus_value = diabetes_mellitus_mapping.get(diabetes_mellitus, '')
    coronary_artery_disease_value = coronary_artery_disease_mapping.get(coronary_artery_disease, '')
    appetite_value = appetite_mapping.get(appetite, '')
    peda_edema_value = peda_edema_mapping.get(peda_edema, '')
    aanemia_value = aanemia_mapping.get(aanemia, '')
    



    user_input = {
    'age': age,
    'blood_pressure': blood_pressure,
    'specific_gravity': specific_gravity,
    'albumin': albumin,
    'sugar': sugar,
    'blood_glucose_random': blood_glucose_random,
    'blood_urea': blood_urea,
    'serum_creatinine': serum_creatinine,
    'sodium': sodium,
    'potassium': potassium,
    'haemoglobin': haemoglobin,
    'packed_cell_volume': packed_cell_volume,
    'white_blood_cell_count': white_blood_cell_count,
    'red_blood_cell_count': red_blood_cell_count,
    'red_blood_cells': red_blood_cells_value,
    'pus_cell': pus_cell_value,
    'pus_cell_clumps': pus_cell_clumps_value,
    'bacteria': bacteria_value,
    'hypertension': hypertension_value,
    'diabetes_mellitus': diabetes_mellitus_value,
    'coronary_artery_disease': coronary_artery_disease_value,
    'appetite': appetite_value,
    'peda_edema': peda_edema_value,
    'aanemia': aanemia_value,
    }

    # code for Prediction
    kidney_disease_diagnosis = ''


    def check_missing_inputs(data):
        return any(value == '' for value in data.values())


    def make_prediction(data):
        df = pd.DataFrame([data])
        X = pd.DataFrame(artifact_4['preprocessing'].transform(df),
                     columns=artifact_4['preprocessing'].get_feature_names_out())
        
        prediction = artifact_4['model'].predict(X)[0]
        decision = 'Kidney Disease Detected' if prediction == 1 else 'No Kidney Disease Detected'
        return decision


    # Button to make prediction
    if st.button('Kidney Disease Test Result'):
        if check_missing_inputs(user_input):
            st.error("Please fill in all the fields.")
        else:
            result = make_prediction(user_input)
            st.success(result)
