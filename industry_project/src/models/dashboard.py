import streamlit as st
import pandas as pd
import requests
import os

# ==========================================
# 1. DASHBOARD CONFIGURATION
# ==========================================
st.set_page_config(page_title="Zeiss Quality Gate", layout="wide")
st.title("Zeiss Predictive Reliability Dashboard")
st.markdown("Automated Quality Gate powered by Bayesian Inference")

# ==========================================
# 2. AUTOMATIC DATA RETRIEVAL
# ==========================================
# Simulating the moment the plusData pipeline drops a new CSV file
# ==========================================
# 2. AUTOMATIC DATA RETRIEVAL
# ==========================================
# Simulating the moment the plusData pipeline drops a new CSV file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data") 

try:
    # Read BOTH files
    df_azure = pd.read_csv(os.path.join(DATA_DIR, 'azure_devops_data.csv'))
    df_security = pd.read_csv(os.path.join(DATA_DIR, 'blackduck_scan_data.csv'))
    
    # Merge them to get all metrics in one place
    df_merged = pd.merge(df_azure, df_security, on='build_id', how='inner')
    
    # Grab the absolute newest row (the bottom one) from the combined data
    latest_build = df_merged.iloc[-1]
    
    # Prepare the payload for the backend API
    metrics = {
        "code_churn": int(latest_build['code_churn']),
        "crit_vulns": int(latest_build['crit_vulns']),
        "test_pass_rate": float(latest_build['test_pass_rate'])
    }
    st.subheader(f"📡 Latest Build Detected: `{latest_build['build_id']}`")
    
except Exception as e:
    st.error(f"Waiting for data from PlusData Storage... (Error: {e})")
    st.stop()
# ==========================================
# 3. ASK THE BACKEND FOR THE SCORE
# ==========================================
# We send the metrics to your FastAPI 'Brain' and wait for the JSON response
API_URL = "http://127.0.0.1:8000/predict"

try:
    response = requests.post(API_URL, json=metrics)
    if response.status_code == 200:
        result = response.json()
    else:
        st.error(f"Backend Error: {response.text}")
        st.stop()
except requests.exceptions.ConnectionError:
    st.error("🚨 Backend API is offline! Please start FastAPI first.")
    st.stop()

# ==========================================
# 4. VISUAL METRICS (Risk & Confidence)
# ==========================================
score = result["Crash_Probability"]
confidence = result["Confidence_Score"]
risk_level = result["Risk_Level"]

col1, col2, col3 = st.columns(3)

col1.metric(
    label="⚠️ Risk Score (Probability of Crash)", 
    value=f"{score:.1%}", 
    delta="High Risk" if risk_level == "High" else "Safe",
    delta_color="inverse"
)

col2.metric(
    label="✅ Reliability / Confidence Score", 
    value=f"{confidence:.1%}"
)

col3.metric(
    label="🛑 System Recommendation", 
    value=result["Recommendation"]
)

st.divider()

# ==========================================
# 5. LEAF STRUCTURE / FEATURE WEIGHTS
# ==========================================
st.subheader("🌿 Feature Impact Analysis (Leaf Structure)")
st.write("Visual representation of how much weight each input parameter carries in deciding the risk score.")

# Note: Bayesian networks use probability distributions, not strict mathematical "weights" like Neural Networks.
# For this prototype visual, we use simulated weights to show the dashboard's capability.
# When you plug in your XGBoost model later, you can replace this with actual `feature_importances_`!

importance_data = pd.DataFrame({
    "Feature": ["Critical Vulnerabilities", "Code Churn", "Test Pass Rate"],
    "Impact Weight (%)": [45, 35, 20] 
}).set_index("Feature")

st.bar_chart(importance_data, color="#005BBB") # Zeiss Blue