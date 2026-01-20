import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# --- 1. SETUP ---
# Set the visual style for the plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

print("Loading data...")
df = pd.read_csv('dengue_data_cleaned_debug.csv')

# --- 2. PREPARE DATA (Re-creating our Logic) ---
# Create the Target Variable (Simulation)
df['Dengue_Label'] = np.where(df['Platelet (cells/cu.mm)'] < 100000, 1, 0)
df['Risk_Category'] = df['Dengue_Label'].map({1: 'High Risk (Severe)', 0: 'Low Risk'})

# Prepare Numeric Columns for Correlation Matrix
# We map Sex to numbers so it shows up in the correlation plot
df['Sex_Code'] = df['Sex'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0}).fillna(0)
numeric_df = df[['Haemoglobin (gm/Dl)', 'Platelet (cells/cu.mm)', 
                 'Red Blood Cell Count (millions/cu.mm)', 
                 'Hematocrit (Packed Cell Volume) (%)', 'Sex_Code', 'Dengue_Label']]

# --- 3. GENERATE GRAPHS ---

# === GRAPH 1: PLATELET DISTRIBUTION ===
print("Generating Graph 1: Platelet Distribution...")
plt.figure(figsize=(10, 6))
# Histogram with a Kernel Density Estimate (KDE) line
sns.histplot(data=df, x='Platelet (cells/cu.mm)', hue='Risk_Category', kde=True, palette='viridis', bins=30)
plt.title('Distribution of Platelet Counts in Patients', fontsize=16)
plt.xlabel('Platelet Count (cells/cu.mm)')
plt.ylabel('Number of Patients')
plt.axvline(100000, color='red', linestyle='--', label='Critical Threshold (100k)')
plt.legend()
plt.savefig('graph_1_platelet_distribution.png')
print(" -> Saved 'graph_1_platelet_distribution.png'")

# === GRAPH 2: CORRELATION HEATMAP ===
print("Generating Graph 2: Correlation Heatmap...")
plt.figure(figsize=(10, 8))
# Calculate correlation matrix
corr = numeric_df.corr()
# Plot heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix (Relationships between Vitals)', fontsize=16)
plt.savefig('graph_2_correlation_matrix.png')
print(" -> Saved 'graph_2_correlation_matrix.png'")

# === GRAPH 3: FEATURE IMPORTANCE ===
print("Generating Graph 3: Feature Importance...")
# We need to train a quick model to get the "Importance" scores
X = numeric_df.drop('Dengue_Label', axis=1)
y = numeric_df['Dengue_Label']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Plot
plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh', color='#2c3e50')
plt.title('Which Factors Determine Dengue Risk?', fontsize=16)
plt.xlabel('Importance Score')
plt.savefig('graph_3_feature_importance.png')
print(" -> Saved 'graph_3_feature_importance.png'")

print("\nAll graphs generated successfully! Check your folder.")