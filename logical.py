from sklearn.metrics import confusion_matrix, accuracy_score

# --- STEP 1: DEFINE THE RULES (The "Common Sense" Logic) ---
def simple_quality_gate(row):
    # Start with a safe score
    risk_score = 0
    
    # Rule 1: High Code Churn is dangerous
    if row['code_churn'] > 200:
        risk_score += 30
    
    # Rule 2: Critical Vulnerabilities are automatic blockers
    if row['crit_vulns'] > 0:
        risk_score += 50  # Huge penalty
        
    # Rule 3: Low Test Pass Rate
    if row['test_pass_rate'] < 0.95:
        risk_score += 20
        
    # --- THE DECISION ---
    # If Risk Score > 40, we predict a CRASH (Failure)
    if risk_score > 40:
        return 1 # Predict Crash
    else:
        return 0 # Predict Safe

# --- STEP 2: APPLY THE RULES ---
df_master['predicted_crash_simple'] = df_master.apply(simple_quality_gate, axis=1)

# --- STEP 3: MEASURE PERFORMANCE (Did we get it right?) ---
# We compare our 'Prediction' vs the 'Actual Truth' (did_crash)
accuracy = accuracy_score(df_master['did_crash'], df_master['predicted_crash_simple'])
tn, fp, fn, tp = confusion_matrix(df_master['did_crash'], df_master['predicted_crash_simple']).ravel()

print(f"📊 SIMPLE MODEL RESULTS")
print(f"-----------------------")
print(f"Overall Accuracy: {accuracy:.1%}")
print(f"-----------------------")
print(f"✅ Correctly Predicted SAFE: {tn} builds")
print(f"✅ Correctly Predicted CRASH: {tp} builds (True Positives)")
print(f"❌ False Alarms (False Positives): {fp} builds (We blocked good code)")
print(f"❌ Missed Crashes (False Negatives): {fn} builds (DANGER! We let bad code through)")

# Show a few examples of where we failed
print("\n--- EXAMPLE OF A MISSED CRASH (False Negative) ---")
missed_crashes = df_master[(df_master['did_crash'] == 1) & (df_master['predicted_crash_simple'] == 0)]
if not missed_crashes.empty:
    display(missed_crashes.head(2)[['build_id', 'code_churn', 'crit_vulns', 'test_pass_rate', 'did_crash', 'predicted_crash_simple']])
else:
    print("Wow! No missed crashes in the top rows.")