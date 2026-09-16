# 🚌 Bus Ticket Demand Forecasting – Time-Aware Predictive Modelling

## 📌 Project Overview

This project builds a machine learning pipeline to forecast route-level bus ticket demand using historical booking and search behaviour.

The objective is to predict the **final seat demand for a journey using information available 15 days before departure**, while maintaining chronological separation between training and validation data.

## 🎯 Problem Statement

Bus ticket demand varies based on route, journey date, location characteristics, booking activity, search behaviour, and seasonal patterns.

The goal of this project is to use historical demand signals to predict:

**Target:** `final_seatcount`

The prediction is made using booking/search information available at **DBD = 15 (15 days before departure)**.

## 📂 Dataset

The project uses four datasets:

**transactions.csv**

* Journey date and booking date
* Days before departure (DBD)
* Source and destination IDs
* Source/destination regions
* Source/destination city tiers
* Cumulative seat bookings
* Cumulative searches

**train.csv**

* Journey date
* Source ID
* Destination ID
* Final seat count

**test.csv**

* Route key
* Journey date
* Source ID
* Destination ID

**sample_submission.csv**

* Required prediction submission format

## 🔍 Data Quality & EDA

Performed:

* Missing-value analysis
* Duplicate detection
* Date conversion and validation
* DBD consistency validation
* Target distribution analysis
* IQR-based outlier investigation
* Booking/search correlation analysis
* Route and calendar-based demand exploration

The target variable showed a strong right-skewed distribution. High-demand observations were retained because they may represent genuine peak-demand journeys rather than data errors.

## ⚙️ Feature Engineering

Created time-aware features using information available before departure.

### Booking & Search Features

* Cumulative bookings at DBD 15
* Cumulative searches at DBD 15
* Bookings at DBD 20
* Searches at DBD 20
* Booking growth from DBD 20 → DBD 15
* Search growth from DBD 20 → DBD 15

### Calendar Features

* Year
* Month
* Week
* Day
* Day of week
* Weekend indicator

### Route Features

* Source city
* Destination city
* Source region
* Destination region
* Source tier
* Destination tier

Categorical variables were transformed using **One-Hot Encoding**.

## ⏳ Time-Aware Validation

Instead of using a random train-test split, chronological validation was used.

Historical observations were used for model training, while **November–December 2024** were kept as an untouched future validation period.

Date-based cross-validation folds were also created for hyperparameter tuning to reduce future-data leakage.

## 🤖 Models Evaluated

The following regression models were compared:

* Linear Regression
* Random Forest Regressor
* Tuned Random Forest
* XGBoost Regressor
* Tuned XGBoost

A log-transformed target was also tested with Linear Regression, but it significantly reduced performance and was therefore not selected.

## 📊 Model Performance

| Model               | Train R² | Validation R² | Validation MAE | Validation RMSE |
| ------------------- | -------: | ------------: | -------------: | --------------: |
| Linear Regression   |    0.400 |         0.342 |         780.01 |         1057.94 |
| Random Forest       |    0.968 |         0.634 |         548.68 |          788.83 |
| Tuned Random Forest |    0.942 |         0.687 |         510.88 |          729.41 |
| XGBoost             |    0.853 |     **0.699** |         511.50 |      **715.05** |
| Tuned XGBoost       |    0.852 |         0.680 |         519.34 |          737.74 |

## 🏆 Final Model

**XGBoost Regressor** was selected as the final model based on its overall validation performance.

### Validation Performance

* **R²:** 0.6995
* **MAE:** 511.50 seats
* **RMSE:** 715.05 seats

The model was retrained using the complete historical training dataset before generating predictions for the future test period.

## 🔎 Model Interpretation

XGBoost feature importance was analyzed to understand which booking, search, route, and calendar variables contributed most to demand predictions.

Prediction-error analysis was also performed across different demand levels to identify where the model produced larger forecasting errors.

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib

## 🔄 Project Workflow

Data Loading
↓
Data Quality Checks
↓
Exploratory Data Analysis
↓
DBD 15 Booking/Search Snapshot
↓
DBD 20 → 15 Momentum Features
↓
Calendar & Route Feature Engineering
↓
One-Hot Encoding
↓
Chronological Train/Validation Split
↓
Date-Based Cross-Validation
↓
Linear Regression
↓
Random Forest + Hyperparameter Tuning
↓
XGBoost + Hyperparameter Tuning
↓
Model Comparison
↓
Feature Importance & Error Analysis
↓
Final Future Demand Prediction

## 📌 Conclusion

The project demonstrates how booking behaviour, search activity, route characteristics, and calendar information can be combined to forecast future bus demand.

Using chronological validation provided a more realistic evaluation than a random split. XGBoost achieved approximately **0.70 validation R²**, demonstrating the ability to capture nonlinear demand patterns while maintaining stronger future-period generalization than the other tested models.
