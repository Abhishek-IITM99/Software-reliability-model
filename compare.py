import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

print("🚀 STARTING MODEL BATTLE ROYALE...")

# --- STEP 1: PREPARE DATA FOR ML (Use Raw Numbers) ---
# We use the raw numbers because ML models handle them better than bins
X = df_master[['code_churn', 'crit_vulns', 'test_pass_rate']]
y = df_master['did_crash']

# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data (Crucial for Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {'Bayesian Network': 0.795} # Your previous result

# --- CONTENDER 1: RANDOM FOREST ---
print("\n🌲 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
results['Random Forest'] = accuracy_score(y_test, rf_preds)

# --- CONTENDER 2: XGBOOST ---
print("🚀 Training XGBoost...")
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
results['XGBoost'] = accuracy_score(y_test, xgb_preds)

# --- CONTENDER 3: NEURAL NETWORK (MLP) ---
print("🧠 Training Neural Network...")
# Simple architecture: 2 hidden layers with 16 and 8 neurons
nn_model = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
nn_model.fit(X_train_scaled, y_train)
nn_preds = nn_model.predict(X_test_scaled)
results['Neural Network'] = accuracy_score(y_test, nn_preds)

# --- FINAL SCOREBOARD ---
print("\n🏆 --- FINAL RESULTS --- 🏆")
df_results = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
df_results = df_results.sort_values(by='Accuracy', ascending=False)
display(df_results)
