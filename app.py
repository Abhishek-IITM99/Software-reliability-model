import streamlit as st
import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MaximumLikelihoodEstimator
import time

# --- 1. SETUP THE PAGE ---
st.set_page_config(page_title="Zeiss Quality Gate", page_icon="🔬", layout="wide")

st.title("🔬 Zeiss SMT: Predictive Quality Gate")
st.markdown("### Probabilistic Reliability Assessment for D100/CMT Software")

# --- 2. TRAIN MODEL (Hidden from User) ---
@st.cache_resource # This makes it fast (caches the model)
def train_model():
    # RE-GENERATE DATA (Or load your CSV)
    # For the app, we generate it on the fly to keep it simple
    np.random.seed(42)
    df = pd.DataFrame({
        'code_churn': np.random.randint(10, 500, 1000),
        'crit_vulns': np.random.choice([0, 1, 5], 1000, p=[0.8, 0.15, 0.05]),
        'test_pass_rate': np.random.uniform(0.90, 1.0, 1000),
        'did_crash': np.random.choice([0, 1], 1000, p=[0.9, 0.1]) # Dummy target
    })
    
    # Discretize
    df['churn_level'] = pd.cut(df['code_churn'], bins=[-1, 50, 200, 10000], labels=['Low', 'Medium', 'High'])
    df['security_status'] = pd.cut(df['crit_vulns'], bins=[-1, 0, 100], labels=['Safe', 'Risky'])
    df['test_quality'] = pd.cut(df['test_pass_rate'], bins=[-1, 0.98, 1.1], labels=['Bad', 'Good'])
    df['crash_prediction'] = df['did_crash'].map({0: 'Stable', 1: 'Crash'})
    
    model_data = df[['churn_level', 'security_status', 'test_quality', 'crash_prediction']]
    
    # Train
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork

    model = BayesianNetwork([
        ('churn_level', 'crash_prediction'), 
        ('security_status', 'crash_prediction'),
        ('test_quality', 'crash_prediction')
    ])
    model.fit(model_data, estimator=MaximumLikelihoodEstimator)
    return VariableElimination(model)

infer = train_model()

# --- 3. THE SIDEBAR (User Inputs) ---
st.sidebar.header("🛠️ Build Parameters")
churn_input = st.sidebar.selectbox("Code Churn Volume", ['Low', 'Medium', 'High'])
sec_input = st.sidebar.selectbox("Security Scan Status", ['Safe', 'Risky'])
test_input = st.sidebar.selectbox("Test Pass Rate", ['Good', 'Bad'])

st.sidebar.markdown("---")
st.sidebar.info("Adjust inputs to see how the Bayesian Network calculates risk in real-time.")

# --- 4. THE MAIN DASHBOARD ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Prediction Engine")
    
    # Run Inference
    prob = infer.query(variables=['crash_prediction'], 
                       evidence={
                           'churn_level': churn_input, 
                           'security_status': sec_input, 
                           'test_quality': test_input
                       })
    
    # Get Crash Probability safely
    try:
        if prob.state_names['crash_prediction'][0] == 'Crash':
            risk_score = prob.values[0]
        else:
            risk_score = prob.values[1]
    except:
        risk_score = 0.0
        
    # DISPLAY THE BIG SCORE
    st.metric(label="Predicted Failure Probability", value=f"{risk_score:.1%}", delta=f"{'-' if risk_score < 0.2 else '+'} Risk Level")
    
    if risk_score < 0.2:
        st.success("✅ **GO STATUS:** Confidence is High. Proceed to Deploy.")
    elif risk_score < 0.5:
        st.warning("⚠️ **CAUTION:** Manual Approval Required.")
    else:
        st.error("🛑 **NO-GO:** High Risk of Failure Detected.")

with col2:
    st.subheader("📊 Live Factor Analysis")
    st.bar_chart({'Risk Factor': risk_score, 'Safety Factor': 1-risk_score})
    st.caption("Visualizing the probability split between Crash vs. Stable.")

# --- 5. AUTOMATED REPORT ---
st.markdown("---")
st.subheader("📝 Automated Decision Log")
st.text(f"Build ID: B-{np.random.randint(1000,9999)}\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\nDecision: {'REJECT' if risk_score > 0.5 else 'APPROVE'}")