# Mobile-Usage-Productivity-Analysis

## Overview
This project analyzes the impact of mobile usage on productivity and sleep behavior using Machine Learning and Power BI visualization techniques.

The project uses Polynomial Regression to predict study hours based on screen time and sleep patterns. It also includes custom-built regression evaluation metrics implemented manually without relying completely on sklearn metrics.

---

## Features

- Data Cleaning and Preprocessing
- Min-Max Normalization
- Polynomial Regression Model
- Custom Evaluation Metrics
  - R² Score
  - MSE
  - RMSE
  - MAE
  - Correlation Coefficient
- Data Visualization using Matplotlib
- Interactive Power BI Dashboard
- Study Hours Prediction System

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Power BI
- Excel

---

## Project Structure

Mobile-Usage-Productivity-Analysis/

├── main.py

├── metrics_module.py

├── Personal_data.xlsx

├── Survey_data.xlsx

├── Power BI dashboard.pbix

├── requirements.txt

├── README.md

├── Data_visualization.png

└── PowerBi_dashboard.png

---

## Machine Learning Model

The project uses Polynomial Regression with degree 2 to capture non-linear relationships between:

- Screen Time
- Sleep Hours
- Study Hours

### Performance Metrics

- R² Score: 0.612
- MSE: 0.021
- RMSE: 0.145

---

## Dashboard Preview

### Python Visualization

![Python Dashboard](Data_visualization.png)

### Power BI Dashboard

![Power BI Dashboard](PowerBi_dashboard.png)

---

## Dataset Information

The project uses:

- Personal behavioral dataset
- Survey dataset containing 500+ records

Dataset columns:

- ScreenTime
- SocialMedia
- StudyHours
- Focus
- SleepHours
- SleepQuality
- PhoneBeforeSleep

---

## Future Improvements

- Real-time data collection
- Web dashboard deployment
- Advanced ML models
- Larger behavioral datasets

---

## Author

Rahul Ubale

AI & DS Student | Machine Learning & Data Analytics Enthusiast