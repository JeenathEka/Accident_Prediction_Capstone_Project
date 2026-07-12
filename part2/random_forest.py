import os
import pandas as pd
import joblib

MODULE_DIR = os.path.abspath(os.path.dirname(__file__))

# Load the trained model and encoders using paths 
rf_model = joblib.load(os.path.join(MODULE_DIR, 'random_forest_model.pkl'))
ohe = joblib.load(os.path.join(MODULE_DIR, 'one_hot_encoder.pkl'))
le = joblib.load(os.path.join(MODULE_DIR, 'label_encoder.pkl'))


def random_forest_model(state_name, city_name, month, day_of_week, weather_conditions):
    
    # Create a DataFrame with the input features
    sample = pd.DataFrame({
        "State Name": [state_name],
        "City Name": [city_name],
        "Month": [month],
        "Day of Week": [day_of_week],
        "Weather Conditions": [weather_conditions]
    })
    
    # encoding
    sample_encoded = ohe.transform(sample)
    
    # make prediction
    pred_encoded = rf_model.predict(sample_encoded)[0]
    
    # decoding
    pred_label = le.inverse_transform([pred_encoded])[0]
    
    return pred_label



if __name__ == "__main__":
    result = random_forest_model(
        state_name="Maharashtra",
        city_name="Pune",
        month="January",
        day_of_week="Monday",
        weather_conditions="Clear"
    )
    print(f"Predicted Accident Location Details: {result}")
