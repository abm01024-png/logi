
import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('delivery_delay.sav')

st.title('Delivery Delay Prediction App')

st.write("Enter the feature values below to predict delivery delay.")

# Define the input fields for each feature
delivery_distance = st.number_input('Delivery Distance', min_value=0.0, value=19.35)
traffic_congestion = st.number_input('Traffic Congestion (1-5)', min_value=1, max_value=5, value=4, step=1)
weather_condition = st.number_input('Weather Condition (1-3, e.g., 1:Good, 2:Moderate, 3:Bad)', min_value=1, max_value=3, value=3, step=1)
delivery_slot = st.number_input('Delivery Slot (1-3)', min_value=1, max_value=3, value=2, step=1)
driver_experience = st.number_input('Driver Experience (years)', min_value=0, value=4, step=1)
num_stops = st.number_input('Number of Stops', min_value=0, value=2, step=1)
vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, value=8, step=1)
road_condition_score = st.number_input('Road Condition Score (1-5)', min_value=1, max_value=5, value=2, step=1)
package_weight = st.number_input('Package Weight (kg)', min_value=0.0, value=15.0)
fuel_efficiency = st.number_input('Fuel Efficiency (km/l)', min_value=0.0, value=12.0)
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=80, step=1)


# Create a DataFrame from the inputs
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error('Delivery is likely to be DELAYED.')
    else:
        st.success('Delivery is likely to be ON TIME.')

    st.subheader('Prediction Probabilities:')
    st.write(f"On Time: {prediction_proba[0][0]:.2f}")
    st.write(f"Delayed: {prediction_proba[0][1]:.2f}")
