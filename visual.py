import matplotlib.pyplot as plt
import networkx as nx
from sklearn.tree import plot_tree
import xgboost as xgb
import seaborn as sns

# Create a canvas for 4 plots
fig, axes = plt.subplots(2, 2, figsize=(20, 15))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# --- 1. WHITE BOX: Bayesian Network ---
axes[0, 0].set_title("1. Bayesian Network (White Box)\nClear Causal Structure", fontsize=14, fontweight='bold', color='green')

# FIX: We calculate the layout explicitly ('circular' ensures no overlap)
pos = nx.circular_layout(model) 
nx.draw(model, pos=pos, with_labels=True, ax=axes[0, 0], 
        node_size=3000, node_color="lightblue", 
        font_size=10, font_weight="bold", arrowsize=20)

# --- 2. GRAY BOX: Random Forest ---
axes[0, 1].set_title("2. Random Forest (Gray Box)\nComplex Flowchart of Rules", fontsize=14, fontweight='bold', color='gray')
# Plotting just the first tree
plot_tree(rf_model.estimators_[0], feature_names=X.columns, 
          class_names=['Stable', 'Crash'], filled=True, ax=axes[0, 1], max_depth=3)

# --- 3. GRAY BOX: XGBoost ---
axes[1, 0].set_title("3. XGBoost (Gray Box)\nOptimized but Abstract Rules", fontsize=14, fontweight='bold', color='gray')
xgb.plot_tree(xgb_model, num_trees=0, ax=axes[1, 0])

# --- 4. BLACK BOX: Neural Network ---
axes[1, 1].set_title("4. Neural Network (Black Box)\nUnreadable Matrix of Weights", fontsize=14, fontweight='bold', color='red')
sns.heatmap(nn_model.coefs_[0], ax=axes[1, 1], cmap="viridis", cbar=True)
axes[1, 1].set_xlabel("Neurons (Hidden Layer)")
axes[1, 1].set_ylabel("Input Features")

plt.show()
