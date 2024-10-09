import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import pickle


final_df = pd.read_csv('Dataset\\finaldata.csv') 


label_encode_columns =["modelYear","Engine","No of Cylinder","Seating Capacity",'ownerNo'] 

one_hot_encode_columns = ["bt","oem","Insurance Validity","transmission",'ft',"city",'Gear Box']

Numerical_columns = ["Mileage","Length","Width","Height","Wheel Base","Kerb Weight",'Max Power','Torque']

def load_objects():
    models = {}
    with open('pikle\\model_xgb.pkl', 'rb') as f:
        models = pickle.load(f)
    with open('pikle\\scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('pikle\\onehot_encoder.pkl', 'rb') as f:
        onehot_encoder = pickle.load(f)
    # Load label encoders for each specific column
    label_encoders = {}
    for col in label_encode_columns:
        with open(f'pikle\\label_encoder_{col}.pkl', 'rb') as f:
            label_encoders[col] = pickle.load(f)
    return models, scaler, onehot_encoder, label_encoders

models, scaler, onehot_encoder, label_encoders = load_objects()

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚘",
    layout="wide",
)

st.markdown("<h1 style='text-align: center;'>🚘 Car Price Prediction</h1>", unsafe_allow_html=True)

Car_Detail, Car_Specs = st.columns(2)



label_encode_columns =["modelYear","Engine","No of Cylinder","Seating Capacity",'ownerNo'] 
one_hot_encode_columns = ["bt","oem","Insurance Validity","Transmission",'ft',"city",'Gear Box']
Numerical_columns = ["Mileage","Length","Width","Height","Wheel Base","Kerb Weight",'Max Power','Torque']

with Car_Detail:
    city = st.selectbox("City", sorted(final_df['city'].unique()))
    ownerNo = st.selectbox("Owner No", sorted(final_df['ownerNo'].unique()))
    fuel_type = st.selectbox("Fuel Type", sorted(final_df['ft'].unique()))
    body_type = st.selectbox("Body Type", final_df['bt'].unique())
    Transmission = st.selectbox("Transmission", sorted(final_df['Transmission'].unique()))
    oem = st.selectbox("Brand", sorted(final_df['oem'].unique())) 
    model_year = st.selectbox("Model Year", sorted(final_df['modelYear'].unique()))
    insurance_validity = st.selectbox("Insurance Validity", sorted(final_df['Insurance Validity'].unique()))
    seating_capacity = st.selectbox("Seating Capacity",sorted(final_df['Seating Capacity'].unique()))
    mileage = st.number_input("Mileage (km/l)", min_value=6, max_value=36, value=15)
    engine = st.selectbox("Engine Capacity (cc)", sorted(final_df['Engine'].unique()))
    


with Car_Specs:
    
    gear_box = st.selectbox("Gear Box", sorted(final_df['Gear Box'].unique()))
    no_of_cylinders = st.selectbox("Number of Cylinders", sorted(final_df['No of Cylinder'].unique()))
    width = st.number_input("Width (mm)", min_value=1000, max_value=3000, value=1800)
    length = st.number_input("Length (mm)", min_value=1000, max_value=6000, value=4000)   
    height = st.number_input("Height (mm)", min_value=1000, max_value=2500, value=1500)
    wheel_base = st.number_input("Wheel Base (mm)", min_value=1000, max_value=4000, value=2700)
    kerb_weight = st.number_input("Kerb Weight (kg)", min_value=500, max_value=4000, value=1500)
    torque = st.number_input("Torque (Nm)", min_value=50, max_value=1000, value=250)
    max_power = st.number_input("Max Power (bhp)", min_value=0, max_value=1000, value=100)
    
                                                                


input_data = {
    'ft' : fuel_type,
    'bt': body_type,
    'ownerNo':ownerNo,
    'oem': oem,
    'modelYear': model_year,
    'Insurance Validity': insurance_validity,
    'Transmission': Transmission,
    'Mileage': mileage,
    'Engine': engine,
    'Max Power': max_power,
    'Torque': torque,
    'No of Cylinder': no_of_cylinders,
    'Length': length,
    'Width': width,
    'Height': height,
    'Wheel Base': wheel_base,
    'Kerb Weight': kerb_weight,
    'Gear Box': gear_box,
    'Seating Capacity': seating_capacity,
    'city': city
    }
    

input_df = pd.DataFrame([input_data])


input_encoded = pd.DataFrame(onehot_encoder.transform(input_df[onehot_encoder.feature_names_in_]), 
    columns=onehot_encoder.get_feature_names_out())


for col in label_encode_columns:
    try:
        if not hasattr(label_encoders[col], 'classes_'):
            raise ValueError(f"The encoder for column '{col}' is not a valid LabelEncoder.")
        input_values = input_df[col].unique()
        
        known_classes = label_encoders[col].classes_
        
        for val in input_values:
            if val not in known_classes:
                # st.warning(f"Value '{val}' in column '{col}' is unseen. Mapping to default.")
                input_df[col].replace(val, known_classes[0], inplace=True)
        
        input_df[col] = label_encoders[col].transform(input_df[col])
    
    except Exception as e:
        st.error(f"Error in encoding column '{col}': {e}")


input_df = input_df.drop(columns=one_hot_encode_columns)

input_df = pd.concat([input_df, input_encoded], axis=1)

input_scaled = scaler.transform(input_df)


if st.button('🚀 Predict'):
    try:
        model = models
        prediction = model.predict(input_scaled)
        predicted_price = prediction[0]
        st.markdown("### **Predicted Price:**")
        st.markdown(f"<h1 style='text-align: center; color: green;'>₹{predicted_price:,.2f}</h1>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")