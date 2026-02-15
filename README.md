# Machine Failure Prediction

## Overview
Predict machine failures using sensor data and machine learning for proactive maintenance. The project includes data analysis, feature engineering, model training, and visualization.

---

## Project Contents
- **data/** – Sensor datasets (`data.csv`, `data (1).csv`)  
- **machine_failure_model.pkl** – Trained ML model  
- **Images/** – Visualizations: correlation heatmap, feature importance, sensor histograms  
- **README.md** – Project documentation  

---

## Requirements
```bash
Python 3.8+
pip install pandas numpy matplotlib seaborn scikit-learn
Usage
import pandas as pd
import pickle

# Load data
df = pd.read_csv('data/data.csv')

# Load model
with open('machine_failure_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(df)
Notes
Binary model files may cause merge conflicts; consider using Git LFS.

Visualizations help understand sensor trends and feature importance.

Author
Kunduru Sneha – GitHub | kundurusneha4@gmail.com


