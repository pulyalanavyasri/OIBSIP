# Task 3 - Car Price Prediction

**Track:** Data Science
**Program:** Oasis Infobyte Summer Internship Program (SIP)

## Objective
Predict the selling price of a used car based on its brand, model, age, mileage, fuel type, and other features.

## Dataset
CarDekho Used Car Dataset (cardekho_dataset.csv), sourced from Kaggle.

## Approach
1. Load and inspect the data
2. Exploratory Data Analysis (distribution of selling price, relationships between features)
3. Feature engineering - drop car_name/model, encode categorical columns (brand, seller_type, fuel_type, transmission_type)
4. Train/test split
5. Train and compare two models: Linear Regression (baseline) and Random Forest Regressor
6. Evaluate using R2, MAE, and RMSE
7. Inspect feature importance

## Results
Random Forest outperformed the Linear Regression baseline. max_power and vehicle_age were the most influential features in predicting selling price.

## Tools Used
Python, pandas, numpy, matplotlib, seaborn, scikit-learn
