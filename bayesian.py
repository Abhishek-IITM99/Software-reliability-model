import pandas as pd
import numpy as np
# --- FIX: IMPORT THE SPECIFIC CLASS THE ERROR ASKED FOR ---
try:
    # Try the new specific class first (as per your error)
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    # Fallback to the standard class if that fails
    from pgmpy.models import BayesianNetwork

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from sklearn.metrics import accuracy_score

print("🚀 Starting Bayesian Network Phase...")

# --- STEP 1: PREPARE DATA (Discretization) ---
df_bayes = df_master.copy()

# 1. Binning Code Churn
df_bayes['churn_level'] = pd.cut(df_bayes['code_churn'], 
                                 bins=[-1, 50, 200, 10000], 
                                 labels=['Low', 'Medium', 'High'])

# 2. Binning Vulnerabilities
df_bayes['security_status'] = pd.cut(df_bayes['crit_vulns'], 
                                     bins=[-1, 0, 100], 
                                     labels=['Safe', 'Risky'])

# 3. Binning Test Pass Rate
df_bayes['test_quality'] = pd.cut(df_bayes['test_pass_rate'], 
                                  bins=[-1, 0.98, 1.1], 
                                  labels=['Bad', 'Good'])

# 4. Target Variable
df_bayes['crash_prediction'] = df_bayes['did_crash'].map({0: 'Stable', 1: 'Crash'})

# Keep only the columns we need
model_data = df_bayes[['churn_level', 'security_status', 'test_quality', 'crash_prediction']]

print("✅ Data Discretized. Example Rows:")
display(model_data.head())

# --- STEP 2: DEFINE THE NETWORK STRUCTURE ---
# We use the class we imported safely above
model = BayesianNetwork([
    ('churn_level', 'crash_prediction'),
    ('security_status', 'crash_prediction'),
    ('test_quality', 'crash_prediction')
])

print("✅ Network Structure Defined.")

# --- STEP 3: TRAIN THE MODEL ---
model.fit(model_data, estimator=MaximumLikelihoodEstimator)

print("✅ Model Trained successfully!")

# --- STEP 4: THE "WHAT-IF" ENGINE (Inference) ---
infer = VariableElimination(model)

def predict_risk(churn, security, test):
    prob = infer.query(variables=['crash_prediction'], 
                       evidence={
                           'churn_level': churn, 
                           'security_status': security, 
                           'test_quality': test
                       })
    # Extract the probability of 'Crash'
    try:
        # Check if the first value corresponds to 'Crash'
        if prob.state_names['crash_prediction'][0] == 'Crash':
            return prob.values[0]
        else:
            return prob.values[1]
    except:
        return 0.0

print("\n🤖 INTERROGATING THE MODEL...")

# Scenario A: The "Perfect Build"
prob_a = predict_risk(churn='Low', security='Safe', test='Good')
print(f"Scenario A (Low Churn + Safe + Good Tests): {prob_a:.1%} Chance of Crash")

# Scenario B: The "Nightmare Build"
prob_b = predict_risk(churn='High', security='Risky', test='Bad')
print(f"Scenario B (High Churn + Risky + Bad Tests): {prob_b:.1%} Chance of Crash")

# Scenario C: The "Tricky One"
prob_c = predict_risk(churn='High', security='Safe', test='Good')
print(f"Scenario C (High Churn + Safe + Good Tests): {prob_c:.1%} Chance of Crash")

# --- STEP 5: MEASURE ACCURACY ---
y_true = model_data['crash_prediction']
y_pred = []

for index, row in model_data.iterrows():
    p_crash = predict_risk(row['churn_level'], row['security_status'], row['test_quality'])
    y_pred.append('Crash' if p_crash > 0.5 else 'Stable')

acc = accuracy_score(y_true, y_pred)
print(f"\n🏆 BAYESIAN MODEL ACCURACY: {acc:.1%}")
