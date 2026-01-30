# --- STEP 1: LOAD THE FILES (In case you restarted) ---
df_azure = pd.read_csv('azure_devops_data.csv')
df_security = pd.read_csv('blackduck_scan_data.csv')
df_incidents = pd.read_csv('machine_incident_logs.csv')

print("Files Loaded.")

# --- STEP 2: MERGE DEV + SECURITY (The "Build Profile") ---
# We match them on 'build_id'
df_master = pd.merge(df_azure, df_security, on='build_id', how='inner')

# --- STEP 3: MERGE INCIDENTS (The "Ground Truth") ---
# We match Azure's 'build_id' with Incident's 'related_build_id'
# We use 'left' join because most builds DON'T have a crash.
df_master = pd.merge(df_master, df_incidents, left_on='build_id', right_on='related_build_id', how='left')

# --- STEP 4: CLEAN UP THE "NON-CRASHES" ---
# If a build didn't crash, the 'downtime_minutes' will be NaN (Empty).
# We replace NaN with 0, meaning "Zero Downtime" (Success).
df_master['downtime_minutes'] = df_master['downtime_minutes'].fillna(0)
df_master['did_crash'] = df_master['downtime_minutes'].apply(lambda x: 1 if x > 0 else 0)

# Drop the duplicate ID column from the incident file to keep it clean
df_master = df_master.drop(columns=['related_build_id', 'incident_id', 'error_log'])

# --- STEP 5: VIEW THE MASTERPIECE ---
print("✅ MASTER TABLE CREATED!")
print(f"Total Rows: {len(df_master)}")
print(f"Total Crashes Found: {df_master['did_crash'].sum()}")
print("\n--- PREVIEW (First 5 Rows) ---")
display(df_master.head()) # 'display()' makes it look pretty in Jupyter