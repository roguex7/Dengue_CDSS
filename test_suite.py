import unittest
import pandas as pd
import numpy as np

# --- RE-DEFINING THE LOGIC TO BE TESTED ---
# (In a real production app, we would import these functions from your main script)

def get_season(month):
    """Maps month number to Season Risk (0-3)"""
    if month in [12, 1, 2]: return 0      # Winter
    elif month in [3, 4, 5]: return 1     # Summer
    elif month in [6, 7, 8, 9]: return 3  # Monsoon (High Risk)
    elif month in [10, 11]: return 2      # Post-Monsoon
    return 0

def extract_symptoms(symptom_text):
    """Parses text to find keywords"""
    symptom_text = str(symptom_text).lower()
    return {
        'has_fever': 1 if 'fever' in symptom_text else 0,
        'has_vomit': 1 if 'vomit' in symptom_text else 0
    }

class TestDengueSystem(unittest.TestCase):

    # --- TEST 1: SEASONALITY LOGIC ---
    def test_season_mapping(self):
        print("\nTesting Season Logic...")
        # Test Case A: July (Month 7) should be Monsoon (High Risk = 3)
        self.assertEqual(get_season(7), 3, "July should be Monsoon (Risk 3)")
        
        # Test Case B: January (Month 1) should be Winter (Low Risk = 0)
        self.assertEqual(get_season(1), 0, "January should be Winter (Risk 0)")
        print(" -> Season Logic Passed ✅")

    # --- TEST 2: SYMPTOM EXTRACTION ---
    def test_symptom_parsing(self):
        print("\nTesting Symptom Extraction...")
        # Test Case A: "High Fever and vomiting"
        input_text = "High Fever and vomiting"
        result = extract_symptoms(input_text)
        self.assertEqual(result['has_fever'], 1, "Should detect 'fever'")
        self.assertEqual(result['has_vomit'], 1, "Should detect 'vomit'")
        
        # Test Case B: "Just a headache" (No fever)
        input_text = "Just a headache"
        result = extract_symptoms(input_text)
        self.assertEqual(result['has_fever'], 0, "Should NOT detect 'fever'")
        print(" -> Symptom Logic Passed ✅")

    # --- TEST 3: DATA CLEANING (SIMULATED) ---
    def test_cleaning_logic(self):
        print("\nTesting Data Cleaning...")
        # Create a tiny fake dataset with the "Not Done" error
        data = {'Platelets': ['150000', 'Not Done', '20000']}
        df = pd.DataFrame(data)
        
        # Apply the fix
        df['Platelets'] = pd.to_numeric(df['Platelets'], errors='coerce')
        median_val = df['Platelets'].median() # Should be median of 150k and 20k -> 85k
        df['Platelets'] = df['Platelets'].fillna(median_val)
        
        # Verify
        self.assertFalse(df['Platelets'].isnull().any(), "There should be no empty values left")
        self.assertEqual(df.loc[1, 'Platelets'], 85000.0, "The 'Not Done' cell should be filled with median")
        print(" -> Cleaning Logic Passed ✅")

if __name__ == '__main__':
    unittest.main()