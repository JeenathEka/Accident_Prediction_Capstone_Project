import os
import random
import sys
import time

import streamlit as st


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) # folder parent path
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from part2.random_forest import random_forest_model
from part3.llm_call import llm_finds_areas


st.set_page_config(page_title="Accident Forecasting", layout="centered")

st.title("Predictive Regional Accident Forecasting Using Machine Learning")
st.markdown("A machine learning system that predicts regional accident hotspots by analyzing state, city, and real-time weather data to improve proactive road safety.")

st.divider()

CITY_OPTIONS = {
    "Andhra Pradesh": ["Tirupati", "Vijayawada", "Vijayawada"],
    "Arunachal Pradesh": ["Some City"],
    "Assam":["Some City"],
    "Bihar":["Some City"],
    "Chandigarh":["Some City"],
    "Chhattisgarh":["Some City"],
    "Delhi":["Dwarka","New Delhi","Rohini"],
    "Goa":["Some City"],
    "Gujarat":["Ahmedabad","Surat","Vadodara"],
    "Haryana":["Some City"],
    "Himachal Pradesh":["Some City"],
    "Jammu and Kashmir":["Some City"],
    "Jharkhand":["Some City"],
    "Karnataka":["Bangalore","Mangalore","Mysore"],
    "Kerala":["Some City"],
    "Madhya Pradesh":["Some City"],
    "Maharashtra":["Mumbai","Pune","Nagpur"],
    "Manipur":["Some City"],
    "Meghalaya":["Some City"],
    "Mizoram":["Some City"],
    "Nagaland":["Some City"],
    "Odisha":["Some City"],
    "Puducherry":["Some City"],
    "Punjab":["Some City"],
    "Rajasthan":["Jaipur","Jodhpur","Udaipur"],
    "Sikkim":["Some City"],
    "Tamil Nadu":["Chennai","Coimbatore","Madurai"],
    "Telangana":["Some City"],
    "Tripura":["Some City"],
    "Uttar Pradesh":["Kanpur","Lucknow","Varanasi"],
    "Uttarakhand":["Some City"],
    "West Bengal":["Durgapur","Kolkata","Siliguri"]
}

state_name = st.selectbox("State", list(CITY_OPTIONS.keys()))

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        city_name = st.selectbox("City", CITY_OPTIONS.get(state_name, []))
        month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September","October","November","December"])

    with col2:
        day_of_week = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday" ,"Sunday"])
        weather_conditions = st.selectbox("Weather conditions", ["Clear", "Foggy", "Hazy", "Rainy", "Stormy"])

    submitted = st.form_submit_button("Click here to Forecast Accidents", use_container_width=True, type="primary")

if submitted:
    if not state_name.strip() or not city_name.strip():
        st.warning("Please enter both state name and city name.")
    else:
        time.sleep(1)
        print()
        print("**********")
        print(state_name)
        print(city_name)
        print(month)
        print(day_of_week)
        print(weather_conditions)
        print("**********")
        print()
        print("ML is predicting, Wait....")
        ml_predicted_type_of_road = random_forest_model(
            state_name=state_name,
            city_name=city_name,
            month=month,
            day_of_week=day_of_week,
            weather_conditions=weather_conditions,
        )
        print("pridiction ----> ", ml_predicted_type_of_road)
        print("LLM calling is in progess....")
        json_object = llm_finds_areas(
            state_name=state_name,
            city_name=city_name,
            prediction=ml_predicted_type_of_road,
        )

        if isinstance(json_object, dict):
            list_of_areas = json_object.get("results", [])
        elif isinstance(json_object, list):
            list_of_areas = json_object
        else:
            list_of_areas = []

        st.divider()
        st.subheader("High-risk accident zones identified within these areas.")
        st.markdown(f"The following target areas have been prioritized for proactive accident mitigation.    State: {state_name}, City: {city_name},  weather:  {weather_conditions} ")

        cols = st.columns(4)
        for i, word in enumerate(list_of_areas):
            cols[i % 4].success(word)

        st.info(f"Total High-risk accident zone areas: {len(list_of_areas)}")