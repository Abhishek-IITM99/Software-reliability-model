import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- CONFIGURATION ---
NUM_BUILDS = 1000
START_DATE = datetime(2025, 1, 1)

print(f"🔄 Generating {NUM_BUILDS} synthetic build records...")

# --- HELPER FUNCTIONS ---
def generate_date(start_date, day_offset):
    return start_date + timedelta(days=day_offset)

# --- 1. GENERATE AZURE DEVOPS DATA (The Code) ---
data_azure = {
    'build_id': [f'B{1000+i}' for i in range(NUM_BUILDS)],
    'timestamp': [generate_date(START_DATE, i//5) for i in range(NUM_BUILDS)], # Approx 5 builds per day
    'code_churn': [],
    'complexity_score': [],
    'test_pass_rate': [],
    'team_name': []
}

teams = ['Smt_Control', 'Smt_Process', 'Smt_Data', 'Legacy_Kernel']

for _ in range(NUM_BUILDS):
    # Simulate: 80% of builds are small changes, 20% are massive refactors
    churn = np.random.choice([np.random.randint(10, 100), np.random.randint(200, 1000)], p=[0.8, 0.2])
    
    # Complexity is linked to Churn (More code = more complex)
    complexity = int(churn / 15) + np.random.randint(5, 20)
    
    # Test Pass Rate (High churn slightly lowers pass rate)
    base_pass = 0.99
    if churn > 300: base_pass = 0.90
    pass_rate = min(1.0, np.random.normal(base_pass, 0.05)) # Add some noise
    
    data_azure['code_churn'].append(churn)
    data_azure['complexity_score'].append(complexity)
    data_azure['test_pass_rate'].append(round(pass_rate, 3))
    data_azure['team_name'].append(random.choice(teams))

df_azure = pd.DataFrame(data_azure)

# --- 2. GENERATE BLACK DUCK DATA (The Security) ---
data_security = {
    'build_id': df_azure['build_id'],
    'crit_vulns': [],
    'high_vulns': [],
    'med_vulns': []
}

for _ in range(NUM_BUILDS):
    # Most builds are safe, some have spikes of vulnerabilities
    crit = np.random.choice([0, 1, 2, 5], p=[0.85, 0.1, 0.04, 0.01])
    high = np.random.choice([0, np.random.randint(1, 10)], p=[0.7, 0.3])
    
    data_security['crit_vulns'].append(crit)
    data_security['high_vulns'].append(high)
    data_security['med_vulns'].append(np.random.randint(0, 5))

df_security = pd.DataFrame(data_security)

# --- 3. GENERATE THE "HIDDEN TRUTH" (Did it actually crash?) ---
# We define a "Failure Probability" based on the inputs so the model has something to find.
# Logic: Risk = (Churn * 0.5) + (Critical Vulns * 20) + (Low Test Rate * 100)

incident_data = []

for i in range(NUM_BUILDS):
    row_az = df_azure.iloc[i]
    row_sec = df_security.iloc[i]
    
    risk_score = 0
    risk_score += (row_az['code_churn'] / 1000) * 30  # Max 30 points for massive churn
    risk_score += (row_sec['crit_vulns'] * 25)        # 25 points per critical bug
    if row_az['test_pass_rate'] < 0.95:
        risk_score += 40                              # Big penalty for failing tests
        
    # Probability of crashing
    crash_prob = min(0.95, risk_score / 100)
    
    # Roll the dice
    did_crash = np.random.choice([0, 1], p=[1-crash_prob, crash_prob])
    
    if did_crash:
        incident_data.append({
            'incident_id': f'INC-{random.randint(10000,99999)}',
            'related_build_id': row_az['build_id'],
            'downtime_minutes': np.random.randint(10, 240),
            'error_log': random.choice(['TIMEOUT_ERR', 'MEMORY_LEAK', 'NULL_POINTER', 'SEG_FAULT'])
        })

df_incidents = pd.DataFrame(incident_data)

# --- 4. SAVE TO FILES ---
df_azure.to_csv('azure_devops_data.csv', index=False)
df_security.to_csv('blackduck_scan_data.csv', index=False)
df_incidents.to_csv('machine_incident_logs.csv', index=False)

print("\n✅ SUCCESS!")
print(f"1. Created 'azure_devops_data.csv' ({len(df_azure)} rows)")
print(f"2. Created 'blackduck_scan_data.csv' ({len(df_security)} rows)")
print(f"3. Created 'machine_incident_logs.csv' ({len(df_incidents)} crashes generated)")
print("You are ready to build the model.")