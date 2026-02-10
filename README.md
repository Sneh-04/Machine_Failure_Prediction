🛠️ Machine Failure Prediction System

A Machine Learning–based predictive maintenance system that analyzes industrial sensor data to predict machine failures in advance and raise preventive alerts. This project demonstrates end-to-end ML workflow including data analysis, model training, visualization, and deployment readiness.

📌 Problem Statement

Unexpected machine failures cause:

Production downtime

High maintenance costs

Safety risks

This project aims to predict machine failure before it happens using sensor data such as temperature, RPM, pressure, and VOC levels.

📊 Dataset

The dataset contains sensor readings collected from industrial machines, including:

Temperature

RPM

Input Pressure

VOC Levels

Failure status (target variable)

Data is stored in the data/ directory.

⚙️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib & Seaborn

Jupyter Notebook

Git & GitHub

🧠 Machine Learning Model

Algorithm: Random Forest Classifier

Why Random Forest?

Handles non-linear relationships well

Robust to noise

Provides feature importance for interpretability

📈 Visualizations

The project includes the following visual insights:

🔹 Feature Importance

Identifies which sensors contribute most to failure prediction.

🔹 Sensor Histograms

Shows distribution and abnormal patterns in sensor readings.

🔹 Correlation Heatmap

Helps understand relationships between sensor variables and failure.

Saved plots:

feature_importance.png

sensor_histograms.png

correlation_heatmap.png

🚨 Preventive Alert System

In addition to prediction, the system generates real-time alerts when sensor values cross safe thresholds, such as:

⚠️ Temperature abnormal

⚠️ RPM abnormal

⚠️ VOC high

⚠️ Input pressure abnormal

This enables early maintenance actions.

📂 Project Structure
Machine_Failure_Prediction/
│
├─ data/
│   └─ data (1).csv
├─ notebook.ipynb
├─ feature_importance.png
├─ sensor_histograms.png
├─ correlation_heatmap.png
├─ predictive_system.py
└─ README.md

▶️ How to Run
Option 1: Jupyter Notebook
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook


Open notebook.ipynb and run all cells.

Option 2: Python Script
python predictive_system.py

✅ Results

Achieved high prediction accuracy

Successfully identified critical sensors affecting machine failure

Reduced false alarms using threshold-based alert logic

🚀 Future Improvements

Deploy as a Streamlit web application

Add real-time sensor data integration

Improve model with XGBoost or LSTM

Add ROC curve & confusion matrix analysis

👤 Author

Sneha
📎 GitHub: https://github.com/Sneh-04

⭐ If you like this project

Give it a star ⭐ and feel free to fork or improve it!
