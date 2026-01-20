import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, mean_absolute_error

def get_season(month):
    if month in [12, 1, 2]: return 0      # Winter (Low Risk)
    elif month in [3, 4, 5]: return 1     # Summer
    elif month in [6, 7, 8, 9]: return 3  # Monsoon (High Risk)
    elif month in [10, 11]: return 2      # Post-Monsoon
    return 0

def main():
    print("Loading and Profiling Data...")
    try:
        df = pd.read_csv('dengue_data_cleaned_debug.csv')
    except FileNotFoundError:
        print("❌ Error: 'dengue_data_cleaned_debug.csv' not found.")
        return None, None, None, None, None, None

    # --- 0. SMART COLUMN FIXER ---
    if 'Age' not in df.columns:
        age_candidates = [c for c in df.columns if 'age' in c.lower()]
        if age_candidates: df.rename(columns={age_candidates[0]: 'Age'}, inplace=True)
        else: df['Age'] = 30
            
    if 'Sex' not in df.columns:
        sex_candidates = [c for c in df.columns if 'sex' in c.lower()]
        if sex_candidates: df.rename(columns={sex_candidates[0]: 'Sex'}, inplace=True)
        else: df['Sex'] = 'Male'

    # --- 1. DATA CLEANING ---
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce').fillna(30)
    df['Sex'] = df['Sex'].astype(str).str.title().str.strip()
    
    # Dates & Season
    df['Date_Obj'] = pd.to_datetime(df['Date of Test & Time of Test'], errors='coerce', dayfirst=True)
    df['Season_Risk'] = df['Date_Obj'].dt.month.fillna(0).astype(int).apply(get_season)
    
    symptom_keywords = ['fever', 'headache', 'pain', 'vomit', 'nausea', 'rash']
    df['Symptoms'] = df['Symptoms'].fillna("").astype(str).str.lower()
    for s in symptom_keywords:
        df[f'Has_{s.capitalize()}'] = df['Symptoms'].apply(lambda x: 1 if s in x else 0)
    
    df['Sex_Code'] = df['Sex'].map({'Male': 1, 'Female': 0}).fillna(0)

    # --- 2. RISK ANALYZER TRAINING ---
    print("Training Risk Model...")
    df['Dengue_Label'] = np.where(
        (df['Platelet (cells/cu.mm)'] < 100000) | 
        (df['Hematocrit (Packed Cell Volume) (%)'] > 50) | 
        (df['Haemoglobin (gm/Dl)'] < 7), 
        1, 0
    )

    clf_features = [
        'Platelet (cells/cu.mm)', 'Haemoglobin (gm/Dl)', 
        'Red Blood Cell Count (millions/cu.mm)', 'Hematocrit (Packed Cell Volume) (%)',
        'Age', 'Sex_Code', 
        'Has_Fever', 'Has_Headache', 'Has_Pain', 'Has_Vomit',
        'Season_Risk'  # <--- NEW: Now determining risk based on season!
    ]
    
    X_clf = df[clf_features].fillna(0)
    y_clf = df['Dengue_Label']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    classifier = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    classifier.fit(X_train_c, y_train_c)
    
    # --- 3. FORECAST ENGINE TRAINING ---
    print("Training Forecast Model...")
    df['Day2_Platelets'] = df['Platelet (cells/cu.mm)']
    df['Is_Recovering'] = np.random.choice([0, 1], size=len(df), p=[0.4, 0.6])
    volatility = np.random.uniform(0.10, 0.25, len(df)) 
    df['Day1_Platelets'] = np.where(df['Is_Recovering'] == 1, df['Day2_Platelets'] / (1 + volatility), df['Day2_Platelets'] * (1 + volatility))
    df['Delta_Day1_Day2'] = df['Day2_Platelets'] - df['Day1_Platelets']
    df['Day3_Platelets'] = df['Day2_Platelets'] + (df['Delta_Day1_Day2'] * 0.9) + np.random.normal(0, 1500, len(df))

    reg_features = [
        'Day1_Platelets', 'Day2_Platelets', 'Delta_Day1_Day2',
        'Haemoglobin (gm/Dl)', 'Red Blood Cell Count (millions/cu.mm)', 
        'Hematocrit (Packed Cell Volume) (%)', 'Age', 'Sex_Code',                              
        'Has_Fever', 'Has_Vomit', 'Has_Pain', 'Has_Headache',
        'Season_Risk' # <--- NEW: Now forecasting based on season!
    ]
    
    X_reg = df[reg_features].fillna(0)
    y_reg = df['Day3_Platelets']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    regressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    regressor.fit(X_train_r, y_train_r)
    
    # --- 4. RETURN EVERYTHING ---
    return classifier, regressor, clf_features, reg_features

if __name__ == "__main__":
    main()
