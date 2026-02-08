from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List

app = FastAPI(title="Weather Prediction API")

# Load model and scaler at startup
try:
    model = joblib.load('weather_model.joblib')
    scaler = joblib.load('scaler.joblib')
except:
    print("Error: Model files not found. Run train_and_save.py first.")

# Define the data structure for the input
class WeatherDay(BaseModel):
    max_temp: float
    min_temp: float
    precip: float
    snow: float
    snow_depth: float

class PredictionInput(BaseModel):
    last_14_days: List[WeatherDay]

@app.get("/")
def home():
    return {"message": "Weather Prediction API is running. Go to /docs for testing."}

@app.post("/predict")
def predict(data: PredictionInput):
    if len(data.last_14_days) != 14:
        raise HTTPException(status_code=400, detail="Exactly 14 days of data required.")
    
    # Convert input to DataFrame-like array
    input_list = [[d.max_temp, d.min_temp, d.precip, d.snow, d.snow_depth] for d in data.last_14_days]
    input_df = pd.DataFrame(input_list, columns=['max_temp', 'min_temp', 'precip', 'snow', 'snow_depth'])
    
    # Replicate feature engineering
    raw_flat = input_df.values.flatten().tolist()
    means = input_df.mean().tolist()
    final_features = np.array(raw_flat + means).reshape(1, -1)
    
    # Scale and Predict
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)
    
    return {
        "predicted_min_temp": round(float(prediction[0]), 2),
        "unit": "Fahrenheit"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)