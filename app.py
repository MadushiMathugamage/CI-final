import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# --- PAGE CONFIG ---
st.set_page_config(page_title="Minneapolis Weather Predictor", layout="wide")

st.title("🌡️ Minneapolis/St. Paul Temperature Predictor")
st.markdown("""
This app predicts the **Minimum Temperature** for the next day based on the last 14 days of weather data.
""")

# --- SIDEBAR: FILE UPLOADS ---
st.sidebar.header("1. Upload Data")
train_file = st.sidebar.file_uploader("Upload train.csv", type="csv")
test_file = st.sidebar.file_uploader("Upload test.csv", type="csv")

# --- HELPER FUNCTIONS ---
def prepare_data(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    for col in ['Precipitation (inches)', 'Snow (inches)', 'Snow Depth (inches)']:
        df[col] = pd.to_numeric(df[col].replace('T', '0.005'), errors='coerce')
    df = df.ffill().bfill()
    doy = df['Date'].dt.dayofyear
    df['cos_day'] = np.cos(2 * np.pi * doy / 365.25)
    df['sin_day'] = np.sin(2 * np.pi * doy / 365.25)
    return df

def get_features(window):
    cols = ['Maximum Temperature degrees (F)', 'Minimum Temperature degrees (F)', 
            'Precipitation (inches)', 'Snow (inches)', 'Snow Depth (inches)']
    return window[cols].values.flatten().tolist() + \
           window[cols].mean().tolist() + \
           [window['cos_day'].iloc[-1], window['sin_day'].iloc[-1]]

# --- MAIN LOGIC ---
if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    st.success("Files uploaded successfully!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Data Preview")
        st.write(train_df.head())
    with col2:
        st.subheader("Test Data Preview")
        st.write(test_df.head())

    # Process Data
    train_clean = prepare_data(train_df)
    test_clean = prepare_data(test_df)

    # Feature Engineering
    X, y = [], []
    for i in range(len(train_clean) - 14):
        X.append(get_features(train_clean.iloc[i : i+14]))
        y.append(train_clean.iloc[i+14]['Minimum Temperature degrees (F)'])
    
    X, y = np.array(X), np.array(y)
    X_test_final = np.array([get_features(test_clean.iloc[i*14 : (i+1)*14]) for i in range(len(test_clean)//14)])

    # Model Selection
    st.sidebar.header("2. Model Settings")
    model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "Ridge Regression", "Ensemble (Avg)"])

    if st.button("🚀 Run Prediction"):
        with st.spinner('Training models and predicting...'):
            # Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_test_scaled = scaler.transform(X_test_final)

            # Initialize and Train
            if model_choice == "Random Forest":
                model = RandomForestRegressor(n_estimators=200, random_state=42)
                model.fit(X, y)
                preds = model.predict(X_test_final)
            
            elif model_choice == "Gradient Boosting":
                model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
                model.fit(X_scaled, y)
                preds = model.predict(X_test_scaled)
            
            elif model_choice == "Ridge Regression":
                model = Ridge(alpha=10.0)
                model.fit(X_scaled, y)
                preds = model.predict(X_test_scaled)
            
            else: # Ensemble
                rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)
                gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42).fit(X_scaled, y)
                preds = (rf.predict(X_test_final) + gb.predict(X_test_scaled)) / 2

            # Display Results
            st.subheader(f"Predictions using {model_choice}")
            res_df = pd.DataFrame({'ID': range(len(preds)), 'Predicted Min Temp (F)': preds})
            

            # Chart
            st.line_chart(res_df.set_index('ID'))
            
            # Download Button
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Submission CSV", csv, "submission.csv", "text/csv")

else:
    st.info("Please upload both `train.csv` and `test.csv` in the sidebar to begin.")   
    
    