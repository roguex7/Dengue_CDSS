import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def get_season(month):
    if month in [12, 1, 2]: return 0
    elif month in [3, 4, 5]: return 1
    elif month in [6, 7, 8, 9]: return 3
    elif month in [10, 11]: return 2
    return 0

def calculate_metrics(y_true, y_pred, set_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n   >>> {set_name} Metrics:")
    print(f"     -> MAE (Avg Error): {mae:.2f}")
    print(f"     -> RMSE:            {rmse:.2f}")
    print(f"     -> R² Score:        {r2:.5f}")

def main():
    print("Loading data for Evaluation...")
    try:
        df = pd.read_csv('dengue_data_cleaned_debug.csv')
    except FileNotFoundError:
        print("❌ Error: 'dengue_data_cleaned_debug.csv' not found.")
        return

    # ==========================================
    # LOGIC REPLICATION (Must match train_model.py)
    # ==========================================
    
    # --- 1. Classifier Prep ---
    df['Dengue_Label'] = np.where(df['Platelet (cells/cu.mm)'] < 100000, 1, 0)
    df['Date_Obj'] = pd.to_datetime(df['Date of Test & Time of Test'], errors='coerce', dayfirst=True)
    df['Season_Risk'] = df['Date_Obj'].dt.month.fillna(0).astype(int).apply(get_season)
    
    symptom_keywords = ['fever', 'headache', 'pain', 'vomit', 'nausea', 'rash']
    df['Symptoms'] = df['Symptoms'].fillna("").astype(str).str.lower()
    for s in symptom_keywords:
        df[f'Has_{s.capitalize()}'] = df['Symptoms'].apply(lambda x: 1 if s in x else 0)
    df['Sex_Code'] = df['Sex'].map({'Male': 1, 'Female': 0, 'M': 1, 'F': 0}).fillna(0)

    clf_features = [
        'Haemoglobin (gm/Dl)', 'Platelet (cells/cu.mm)', 
        'Red Blood Cell Count (millions/cu.mm)', 'Hematocrit (Packed Cell Volume) (%)',
        'Sex_Code', 'Has_Fever', 'Has_Headache', 'Has_Pain', 'Has_Vomit'
    ]
    X_clf = df[clf_features].fillna(0)
    y_clf = df['Dengue_Label']

    # --- 2. Regressor Prep (Advanced Physics) ---
    df['Day2_Platelets'] = df['Platelet (cells/cu.mm)']
    
    # Same Exponential Logic as Production Script
    df['Is_Recovering'] = np.random.choice([0, 1], size=len(df), p=[0.4, 0.6])
    volatility = np.random.uniform(0.10, 0.25, len(df)) 
    
    df['Day1_Platelets'] = np.where(
        df['Is_Recovering'] == 1,
        df['Day2_Platelets'] / (1 + volatility), 
        df['Day2_Platelets'] * (1 + volatility)
    )
    
    df['Delta_Day1_Day2'] = df['Day2_Platelets'] - df['Day1_Platelets']
    
    friction = 0.9 
    df['Day3_Platelets'] = df['Day2_Platelets'] + (df['Delta_Day1_Day2'] * friction)
    
    noise = np.random.normal(0, 1500, len(df))
    df['Day3_Platelets'] = df['Day3_Platelets'] + noise

    reg_features = [
        'Day1_Platelets', 'Day2_Platelets', 'Delta_Day1_Day2',
        'Haemoglobin (gm/Dl)', 'Hematocrit (Packed Cell Volume) (%)',
        'Has_Fever', 'Has_Vomit', 'Has_Pain'
    ]
    X_reg = df[reg_features].fillna(0)
    y_reg = df['Day3_Platelets']

    # ==========================================
    # EVALUATION REPORT (Train vs Test)
    # ==========================================
    print("\n📊 --- ADVANCED MODEL EVALUATION --- 📊")
    
    # 1. Evaluate Classifier
    print("\n1️⃣  RISK CLASSIFIER (RandomForest)")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    # Matching params
    classifier = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    classifier.fit(X_train_c, y_train_c)
    
    acc_train = classifier.score(X_train_c, y_train_c)
    acc_test = classifier.score(X_test_c, y_test_c)
    print(f"   -> Training Accuracy: {acc_train*100:.2f}%")
    print(f"   -> Testing Accuracy:  {acc_test*100:.2f}%")

    # 2. Evaluate Regressor
    print("\n2️⃣  FORECAST ENGINE (GradientBoosting - Optimized)")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    # Matching params
    regressor = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    regressor.fit(X_train_r, y_train_r)
    
    y_pred_train = regressor.predict(X_train_r)
    y_pred_test = regressor.predict(X_test_r)
    
    calculate_metrics(y_train_r, y_pred_train, "TRAINING SET")
    calculate_metrics(y_test_r, y_pred_test, "TESTING SET")
    
    print("\n------------------------------------------------")
    print("✅ Evaluation Complete.")

if __name__ == "__main__":
    main()