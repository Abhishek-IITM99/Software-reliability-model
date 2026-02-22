import pandas as pd
import numpy as np
import os

# --- FIX FOR SPECIFIC PGMPY VERSIONS ---
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# 🌟 GLOBAL VARIABLE: This holds the trained model in memory so it doesn't retrain!
infer_engine = None

def train_bayesian_model():
    """Runs ONCE when the API server boots up or when testing locally."""
    global infer_engine
    print("🚀 Starting Bayesian Network Training...")
    
    # --- LOCATE THE DATA ---
    # This automatically looks for a 'data' folder right next to this python file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data") 
    
    try:
        # Load the files
        print(f"📂 Looking for data in: {DATA_DIR}")
        df_azure = pd.read_csv(os.path.join(DATA_DIR, 'azure_devops_data.csv'))
        df_security = pd.read_csv(os.path.join(DATA_DIR, 'blackduck_scan_data.csv'))
        df_incidents = pd.read_csv(os.path.join(DATA_DIR, 'machine_incident_logs.csv'))
        print("✅ Files Loaded.")
    except FileNotFoundError as e:
        print(f"⚠️ Error: CSV files not found. Make sure you have a 'data' folder next to this script containing the 3 CSVs.\nDetails: {e}")
        return False

    # --- MERGE DATA ---
    df_master = pd.merge(df_azure, df_security, on='build_id', how='inner')
    df_master = pd.merge(df_master, df_incidents, left_on='build_id', right_on='related_build_id', how='left')
    
    # Clean up non-crashes
    df_master['downtime_minutes'] = df_master['downtime_minutes'].fillna(0)
    df_master['did_crash'] = df_master['downtime_minutes'].apply(lambda x: 1 if x > 0 else 0)

    # --- BINNING (Discretization) ---
    df_bayes = df_master.copy()
    
    df_bayes['churn_level'] = pd.cut(df_bayes['code_churn'], 
                                     bins=[-1, 50, 200, 10000], 
                                     labels=['Low', 'Medium', 'High'])
                                     
    df_bayes['security_status'] = pd.cut(df_bayes['crit_vulns'], 
                                         bins=[-1, 0, 100], 
                                         labels=['Safe', 'Risky'])
                                         
    df_bayes['test_quality'] = pd.cut(df_bayes['test_pass_rate'], 
                                      bins=[-1, 0.98, 1.1], 
                                      labels=['Bad', 'Good'])
                                      
    df_bayes['crash_prediction'] = df_bayes['did_crash'].map({0: 'Stable', 1: 'Crash'})
    
    model_data = df_bayes[['churn_level', 'security_status', 'test_quality', 'crash_prediction']]

    # --- TRAIN THE MODEL ---
    model = BayesianNetwork([
        ('churn_level', 'crash_prediction'),
        ('security_status', 'crash_prediction'),
        ('test_quality', 'crash_prediction')
    ])
    
    model.fit(model_data, estimator=MaximumLikelihoodEstimator)
    
    # Save the trained engine to the global variable
    infer_engine = VariableElimination(model)
    print("✅ Model Trained and cached in memory!")
    return True


def get_bayesian_prediction(code_churn, crit_vulns, test_pass_rate):
    """Called by the API whenever a new software build needs to be checked."""
    if infer_engine is None:
        print("⚠️ Model not trained yet!")
        return -1.0 
        
    # 1. Convert the raw incoming numbers into your Bins
    churn = 'High' if code_churn > 200 else ('Medium' if code_churn > 50 else 'Low')
    security = 'Risky' if crit_vulns > 0 else 'Safe'
    test = 'Good' if test_pass_rate > 0.98 else 'Bad'

    # 2. Ask the trained model for the probability
    prob = infer_engine.query(
        variables=['crash_prediction'], 
        evidence={'churn_level': churn, 'security_status': security, 'test_quality': test}
    )
    
    # 3. Extract and return the probability of a 'Crash'
    try:
        if prob.state_names['crash_prediction'][0] == 'Crash':
            return float(prob.values[0])
        else:
            return float(prob.values[1])
    except Exception:
        return 0.0


# ==========================================
# --- LOCAL TESTING TRIGGER ---
# ==========================================
# This block ONLY runs if you execute this file directly in the terminal.
if __name__ == "__main__":
    print("\n--- 🧪 RUNNING LOCAL TEST ---")
    
    # 1. Trigger the training
    success = train_bayesian_model()
    
    if success:
        print("\n--- 🎯 TESTING PREDICTION ENGINE ---")
        
        # Scenario A: Feed it a fake "Safe" build
        safe_score = get_bayesian_prediction(code_churn=20, crit_vulns=0, test_pass_rate=0.99)
        print(f"Scenario A (Low Churn, 0 Vulns, 99% Tests) -> Probability of Crash: {safe_score:.1%}")
        
        # Scenario B: Feed it a fake "Risky" build
        risky_score = get_bayesian_prediction(code_churn=500, crit_vulns=2, test_pass_rate=0.85)
        print(f"Scenario B (High Churn, 2 Vulns, 85% Tests) -> Probability of Crash: {risky_score:.1%}")