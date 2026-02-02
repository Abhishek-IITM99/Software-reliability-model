import matplotlib.pyplot as plt
import seaborn as sns

print("🎲 Starting Monte Carlo Simulation...")

# --- CONFIGURATION ---
NUM_SIMULATIONS = 1000
# We simulate a "Borderline" build to see how stable the model is
# Scenario: Medium Churn, No Critical Bugs, but Tests are just "Okay"
simulated_churn = 'Medium'
simulated_security = 'Safe'
simulated_test = 'Good'

risk_distribution = []

for i in range(NUM_SIMULATIONS):
    # We add "Noise" to the model's confidence
    # (Simulating that our input data might be slightly wrong)
    
    # 1. Get the base prediction
    base_prob = predict_risk(simulated_churn, simulated_security, simulated_test)
    
    # 2. Add uncertainty (Standard Deviation of 5%)
    # In real life, sensor data or logs have noise.
    noisy_prob = np.random.normal(base_prob, 0.05) 
    
    # Clip result to stay between 0% and 100%
    noisy_prob = max(0.0, min(1.0, noisy_prob))
    
    risk_distribution.append(noisy_prob)

# --- VISUALIZATION ---
plt.figure(figsize=(10, 6))
sns.histplot(risk_distribution, kde=True, color='green', bins=30)
plt.axvline(np.mean(risk_distribution), color='red', linestyle='dashed', linewidth=2, label=f'Mean Risk: {np.mean(risk_distribution):.1%}')

plt.title(f'Monte Carlo Risk Simulation ({NUM_SIMULATIONS} runs)', fontsize=14)
plt.xlabel('Predicted Probability of Crash', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)

print("✅ Simulation Complete. Displaying Graph...")
plt.show()