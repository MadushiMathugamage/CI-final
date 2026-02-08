import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

def train():
    # Load and Preprocess
    df = pd.read_csv('train.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    for col in ['Precipitation (inches)', 'Snow (inches)', 'Snow Depth (inches)']:
        df[col] = pd.to_numeric(df[col].replace('T', '0.005'), errors='coerce')
    df = df.ffill().bfill()
    
    # Feature Engineering (Sliding Window)
    X, y = [], []
    cols = ['Maximum Temperature degrees (F)', 'Minimum Temperature degrees (F)', 
            'Precipitation (inches)', 'Snow (inches)', 'Snow Depth (inches)']
    
    for i in range(len(df) - 14):
        window = df.iloc[i : i+14]
        # Features: Flattened 14 days + means
        feats = window[cols].values.flatten().tolist() + window[cols].mean().tolist()
        X.append(feats)
        y.append(df.iloc[i+14]['Minimum Temperature degrees (F)'])
    
    # Scale and Train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)
    
    # Save the model and scaler
    joblib.dump(model, 'weather_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model and Scaler saved successfully!")

if __name__ == "__main__":
    train()