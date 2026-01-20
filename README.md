# 🫁 Pneumonia Detection - Clinical Decision Support System (CDSS)

**A full-stack AI solution transforming raw medical imaging data into a real-time diagnostic tool for radiologists.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pneumoniacdss.streamlit.app/)


## 📌 Executive Summary
Pneumonia remains a critical global health challenge. This project implements an end-to-end **Clinical Decision Support System (CDSS)** that uses the **YOLOv5** deep learning architecture to detect and localize pulmonary opacities in Chest X-rays. Unlike standard classifiers, this system provides precise bounding box coordinates and confidence scores, offering a transparent "second opinion" for medical professionals.

---

## 🏗️ Phase 1: Data Engineering & Preprocessing

### 1. Handling Raw DICOM Data
The project utilized the **RSNA Pneumonia Detection Challenge** dataset. The raw medical images were in **DICOM (Digital Imaging and Communications in Medicine)** format, which is standard for hospitals but incompatible with most rapid-training AI pipelines.

* **Challenge:** DICOM files contain complex metadata and high-bit-depth pixel arrays.
* **Solution:** I engineered a preprocessing pipeline using `pydicom` and `OpenCV` to extract the pixel data, normalize it, and convert thousands of images into high-quality **PNG** format.
<img width="1703" height="885" alt="Screenshot 2026-01-18 205606" src="https://github.com/user-attachments/assets/48d34b30-1408-4de6-9397-9ef2c3f13eb0" />
<img width="1692" height="880" alt="Screenshot 2026-01-18 205751" src="https://github.com/user-attachments/assets/15909b81-c900-4f4c-af49-c5efc731221b" />

### 2. Creating a Balanced Dataset (The 500/500 Split)
To prevent the model from becoming biased (e.g., guessing "Healthy" every time because most people are healthy), I curated a strictly balanced training set.

* **Positive Cases (500+):** Images confirmed to have pneumonia opacities.
* **Negative Cases (500-):** Images confirmed as "Normal" or "No Lung Opacity / Not Pneumonia".
* **Label Linking:** I mapped the RSNA CSV clinical labels to standard YOLO `.txt` annotation files, ensuring every image was perfectly linked to its ground-truth bounding box.
<img width="827" height="616" alt="Screenshot 2026-01-18 210030" src="https://github.com/user-attachments/assets/e4156455-b54b-43f8-8999-b53052e6e27b" />
<img width="915" height="698" alt="Screenshot 2026-01-18 210123" src="https://github.com/user-attachments/assets/4e4f20c3-8162-41d2-a093-af100788789b" />

---

## 🧠 Phase 2: Model Training (YOLOv5)

With the data prepared, I configured a custom **YOLOv5s (Small)** architecture, chosen for its balance of inference speed and detection accuracy—critical for clinical settings.

* **Configuration:** 50 Epochs, Batch Size 8.
* **Environment:** Trained on a local GPU environment (`rpnem`).
* **Outcome:** The training process optimized the weights over 50 iterations, saving the model with the highest Mean Average Precision (mAP) as `best.pt`.
<img width="692" height="594" alt="Screenshot 2026-01-18 210301" src="https://github.com/user-attachments/assets/4639ba5c-4bed-4a65-87c3-8536e5434593" />
<img width="663" height="348" alt="Screenshot 2026-01-18 210408" src="https://github.com/user-attachments/assets/2cc82c77-aa95-406d-95e0-be5de3a24884" />


### Training Metrics
The model's performance was tracked using loss curves and validation metrics to ensure it wasn't overfitting to the training data.
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/4f885277-7240-420d-8088-a361b134a5fa" />

---

## 💻 Phase 3: The Clinical Web Application (CDSS)

The final deliverable is a **Streamlit** web application that serves the model to end-users. This interface turns the raw code into a usable medical product.

### 1. The Inference Interface
Radiologists can upload standard PNG/JPG X-rays. The system runs inference in milliseconds (using `@st.cache_resource`) and renders red bounding boxes around suspected pneumonia regions.
<img width="1919" height="969" alt="Screenshot 2026-01-18 005911" src="https://github.com/user-attachments/assets/744ed0d6-2606-418c-a5ab-c128cd7499ee" />
<img width="1545" height="1787" alt="image" src="https://github.com/user-attachments/assets/7c821641-8824-4fbc-b8de-af83fd427eeb" />
<img width="1545" height="2021" alt="image" src="https://github.com/user-attachments/assets/91d038bf-2a73-429a-acf2-8512986be4e1" />


### 2. Clinical Analysis & Reporting
Beyond just visuals, the app generates a detailed **Clinical Report**.
* **Confidence Thresholding:** A dynamic slider (0.10 - 1.0) allows doctors to adjust sensitivity.
<img width="290" height="301" alt="Screenshot 2026-01-18 211329" src="https://github.com/user-attachments/assets/1a599419-df00-42e9-b882-e4e3c7d0c939" />

* **Data Table:** Displays exact spatial coordinates (`xmin`, `ymin`, `xmax`, `ymax`) for every detection.
* **Export:** Findings can be downloaded as a CSV for integration with Electronic Health Records (EHR).
<img width="1372" height="422" alt="image" src="https://github.com/user-attachments/assets/fbf0354b-38a0-4ed8-8854-d78f64487c41" />

---

## 📊 Final Results Summary
Below is a verification grid showing the model's performance on unseen validation data, demonstrating its ability to detect pneumonia across different patient anatomies.
<img width="5613" height="3556" alt="portfolio_results_grid" src="https://github.com/user-attachments/assets/477494fc-6498-4ac4-84ff-03a7c4af7ae8" />

---

## 🛠️ Technical Stack
* **Language:** Python 3.9+
* **Deep Learning:** PyTorch, YOLOv5
* **Web Framework:** Streamlit
* **Image Processing:** OpenCV, Pydicom, PIL
* **Data Handling:** Pandas, NumPy

## ⚙️ How to Run Locally
1.  Clone the repository.
2.  Dataset link:
    ```bash
    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Launch the application:
    ```bash
    streamlit run app.py
    ```

## 🚀 Deployment
This project is deployed using a **CI/CD pipeline** with **Streamlit Community Cloud**.
- **Source:** The app is directly connected to this GitHub repository.
- **Updates:** Pushing changes to the `main` branch automatically triggers a redeploy.
- **Environment:** Linux-based cloud runner with `libgl1` dependencies managed via `packages.txt`.

🔗 **Live Demo:** [Click here to launch the CDSS](https://pneumoniacdss.streamlit.app/)
---
**Developed by:** Annant R Gautam
**Dataset Credit:** Radiological Society of North America (RSNA)
