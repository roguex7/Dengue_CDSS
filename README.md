# 🦟 Dengue CDSS: AI-Powered Clinical Decision Support System

**A Dual-Engine AI System** designed to assist doctors in managing Dengue cases. It features a **Risk Analyzer** for immediate triage and a physics-aware **Trajectory Forecaster** to predict platelet trends 24 hours in advance.

---
## 📖 Executive Summary

Dengue management is a race against time. The critical factor often isn't just the current platelet count, but the **velocity of decay**. A patient with 120,000 platelets dropping fast is in more danger than a stable patient at 80,000.
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://denguecdss.streamlit.app/)


This CDSS (Clinical Decision Support System) solves this by:

1. **Diagnosing Severity:** Using a **Random Forest Classifier** to identify high-risk patients based on multi-factor vitals (Hb, HCT, Platelets).
2. **Predicting the Future:** Using a **Gradient Boosting Regressor** to forecast the next day's platelet count based on the patient's biological momentum.

<img width="1799" height="237" alt="Screenshot 2026-01-20 152013" src="https://github.com/user-attachments/assets/d0bf5d4a-4a50-4fc5-939b-85065ee226bd" />
<img width="501" height="502" alt="b29db3484b14025bc6ff41d8018c1505ceeb119a0008ae8761e0d8ed" src="https://github.com/user-attachments/assets/fb8fd320-cd18-4abc-ae18-d84454c59d70" />

---
## 💻 The User Interface (Evidence of Functionality)

The application utilizes a clean, toggle-based interface to keep doctors focused on the task at hand.

### 1. Navigation Hub

A focused dashboard allowing instant switching between diagnostic and prognostic tools without losing patient context.

<img width="952" height="207" alt="image" src="https://github.com/user-attachments/assets/87b89bb1-424e-4a6e-aa61-bf3899bd0bd4" />

### 2. Risk Analyzer (Diagnosis)

The system evaluates **Platelets, Hematocrit, Haemoglobin, Age, and Symptoms**. Unlike simple calculators, it uses a "Multi-Factor Truth" logic.

* **Evidence of Logic:** Below, the system correctly identifies a stable patient despite low-ish platelets because their Hematocrit and Hemoglobin are within safe ranges.

<img width="1920" height="1582" alt="image" src="https://github.com/user-attachments/assets/f51dfe54-69cf-4bcc-8931-848e85bf8d05" />
<img width="1920" height="1582" alt="image" src="https://github.com/user-attachments/assets/5fb5937e-8b30-4207-b088-497b20c42f10" />



### 3. Trajectory Forecasting (Prognosis)

Using **Momentum Physics**, the AI calculates the *velocity* of change between Day-0 and Day-1 to project the Day-3 outcome.

* **Visualization:** Custom Matplotlib graphs with **floating annotations** ensure doctors can read values without gridline interference.

<img width="1920" height="2087" alt="image" src="https://github.com/user-attachments/assets/0e8667d3-4b87-41e3-8825-ab4565b161c9" />
<img width="1920" height="2087" alt="image" src="https://github.com/user-attachments/assets/38bebeea-aea4-4532-b94b-609909e50279" />

---
## 🧠 Dual-Engine AI Architecture
We moved beyond simple "if/else" guidelines by training two distinct Scikit-Learn models on validated patient history.

Engine A: The Risk Analyzer (Classification)
<img width="908" height="810" alt="Screenshot 2026-01-20 150551" src="https://github.com/user-attachments/assets/ccf15d96-5159-44cb-bf70-e7a9e7efbcbd" />

Algorithm: Random Forest Classifier (n_estimators=200).

Logic: Acts as a "Council of Doctors," voting on risk based on Platelets, Hematocrit, Haemoglobin, Age, and Symptoms.
<img width="901" height="916" alt="Screenshot 2026-01-20 150754" src="https://github.com/user-attachments/assets/4e512519-d57a-43f0-a510-773b2c7a1714" />

Safety Protocol: Implements a "Multi-Factor Truth" logic. It flags high risk if any critical parameter (e.g., Hematocrit > 50%) is breached, ensuring 100% sensitivity to "Hidden Shock" cases.

Engine B: The Forecast Engine (Regression)
<img width="883" height="740" alt="Screenshot 2026-01-20 150636" src="https://github.com/user-attachments/assets/6f12c0df-65e1-4ffc-b1ad-5254269315e1" />

Algorithm: Gradient Boosting Regressor.

Innovation: "Momentum Physics." Instead of treating vitals as static numbers, the model calculates the Velocity (Delta) between Day-0 and Day-1. It uses this speed to project the trajectory for Day-3.

---
## 🛡️ Robustness Factors: Engineering Safety

Medical AI cannot be fragile. This project implements strict **Guardrails** and **Input Validation** to ensure data integrity before it ever reaches the AI "Brain."

### 1. Biological Reality Checks

The system prevents "fat-finger" errors (typos) by enforcing biological limits on inputs.

* **Proof 1 (Hematocrit):** Rejecting values that are physiologically impossible (e.g., ensuring percentages make sense).
  If a user types 95% for Hematocrit, that isn't just 'high'—it is biologically impossible (at that concentration, blood is essentially a solid paste with no plasma to flow, making circulation impossible).         Allowing it would just confuse the AI model.
<img width="577" height="192" alt="image" src="https://github.com/user-attachments/assets/6035c489-7886-4a72-b4e8-f2bf68bc31a4" />


* **Proof 2 (RBC Count):** Flagging RBC counts that exceed human limits, preventing skew in the prediction model.
  If a user types 41.5 for RBC count, that isn't just "high"—it is biologically impossible (human blood turns into solid sludge around 12-14). Allowing it would just confuse the AI model.
<img width="601" height="168" alt="image" src="https://github.com/user-attachments/assets/19b3768d-fa06-4e53-bcb4-e7aadd1675d4" />

---
## ⚙️ The "Dirty Truth" of Data Processing (ETL Pipeline)

One of the biggest challenges in this project was the **Real-World Training Dataset**. Hospital data is notoriously messy. I built a **Self-Healing ETL Pipeline** to handle this.

---
## 🧠 Why Use a "Train Dataset" Instead of Hard Rules?

You might ask: *"Why train an AI? Why not just write code that says `if platelets < 100k, then High Risk`?"*

**The Answer:** Biology is non-linear.

1. **Complex Interactions:** A platelet count of 90k is "safe" for a 20-year-old male but "dangerous" for a 70-year-old female with high Hematocrit. Hard-coded rules miss these nuances.
2. **Pattern Recognition:** By training on a dataset of real patients, the **Random Forest** and **Gradient Boosting** models learn the *hidden correlations* between symptoms (like Vomiting) and recovery speed that aren't obvious in standard guidelines.
3. **Adaptability:** As we feed the model more patient history, it gets smarter at predicting "outlier" cases that defy standard textbook definitions.

---

## 🚀 Installation & Usage

### Prerequisites

* Python 3.10+
* VS Code (Recommended)

### Step 1: Clone & Setup

```bash
git clone https://github.com/roguex7/Dengue_CDSS.git
cd dengue-cdss-ai

```

### Step 2: Activate Environment

```bash
python -m venv .denv
# Windows:
.denv\Scripts\activate
# Mac/Linux:
source .denv/bin/activate

```

### Step 3: Install & Run

```bash
pip install -r requirements.txt
streamlit run app.py

```
## 🚀 Deployment & CI/CD Pipeline

This project is fully deployed as a live web application using **Streamlit Community Cloud**, leveraging a continuous deployment pipeline linked directly to this GitHub repository.

### ☁️ Architecture
* **Infrastructure:** Serverless Linux environment hosted on Streamlit Cloud.
* **Continuous Deployment:** The application enables **"Hot Reloading."** Any code change pushed to the `main` branch automatically triggers a rebuild of the cloud environment, ensuring the live tool is always up-to-date with the latest algorithms.
* **Dependency Management:** The cloud builder automatically provisions the environment using `requirements.txt` to install Scikit-Learn, Pandas, and custom logic libraries.

### 🔗 Live Demo
Access the fully functional CDSS here:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://denguecdss.streamlit.app/)

*(Note: The application performs a "cold boot" if inactive for a while. Please allow 10-20 seconds for the AI models to load into memory upon first access.)*
---

## 👤 Author

**Annant R Gautam**

* **Role:** Full Stack Data Scientist
* **Specialty:** Clinical AI & Predictive Modeling
* **Tech Stack:** Python, Scikit-Learn, Streamlit, Pandas
