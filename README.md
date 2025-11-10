# ds20-covid-19
# 🧬 COVID-19 Mortality Modeling with Machine Learning  

## **Overview**  
This project applies **machine learning regression models** to analyze and predict **daily COVID-19 mortality** across the **WHO European Region (Jan–Jul 2020)**.  
By combining epidemiological, mobility, economic, and demographic data, the study explores which factors most strongly influenced pandemic outcomes during the first wave — and how policy timing and population structure shaped mortality patterns.  

---

## **Objectives**
- Model daily **COVID-19 mortality (New Deaths)** during the early pandemic.  
- Identify **drivers of mortality** across health, policy, and socioeconomic dimensions.  
- Demonstrate **machine learning as a predictive and analytical tool** for epidemic dynamics.  

---

## **Data Sources**

| Source | Files | Description |
|--------|-------|-------------|
| **Kaggle COVID-19 Dataset** | `covid_19_clean_complete.csv`, `day_wise.csv`, `worldometer_data.csv` | Core pandemic indicators — confirmed, recovered, active, deaths. |
| **Our World in Data (OWID)** | `google_mobility.csv`, `gov_policy.csv`, `tracking_r.csv` | Policy measures, mobility trends, and reproduction rate (R). |
| **World Bank Open Data** | `worldbank_gdp.csv`, `worldbank_healthcare_percent.csv`, `worldbank_population.csv`, etc. | Socioeconomic and demographic features (GDP, healthcare spending, population age structure). |

📆 **Timeframe:** January 22 – July 27, 2020  
🌍 **Region:** WHO European Region (excluding microstates and territories with missing data)  

---

## **Workflow**

1. **Data Preparation**  
   - Merged datasets by date and country.  
   - Handled missing data via logical extrapolation, baseline imputation, and **KNN imputer**.  
   - Normalized metrics (Active, Recovered, Tested) **per 100 cases** to reduce noise.  

2. **Exploratory Data Analysis (EDA)**  
   - Correlation matrices, AutoViz profiling, and regional comparison plots.  
   - Identified relationships between demographics, economics, and health outcomes.  

3. **Feature Engineering**  
   - **Lagged Features:** 14- and 30-day delays to model delayed policy/mobility effects.  
   - **Econ_Age_Interaction:** Combined economic and demographic dimensions for better context.  
   - **GeoCluster:** Grouped countries by regional and cultural proximity.  
   - Removed cumulative and perfectly correlated variables to avoid leakage.  

4. **Model Selection & Evaluation**  
   - Algorithms tested:  
     `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`,  
     `GradientBoostingRegressor`, `AdaBoostRegressor`, `SVR`, `XGBRegressor`  
   - Metrics: **MSE**, **RMSE**, **MAE**, **R²**  

| Model | RMSE | MAE | R² | Summary |
|--------|------|------|----|---------|
| Linear Regression | 70.45 | 34.31 | 0.47 | Baseline; fails to capture non-linearity. |
| Decision Tree | 40.33 | 8.85 | 0.83 | Captures patterns but overfits. |
| Random Forest | **30.59** | **7.54** | **0.90** | Excellent generalization. |
| Gradient Boosting | 32.38 | 8.96 | 0.89 | Balanced bias–variance tradeoff. |
| **XGBoost** | **30.81** | **7.45** | **0.90** | Best overall performance and interpretability. |

---

## **Hyperparameter Tuning**

GridSearchCV (3-fold, 50 candidate sets) optimized XGBoost parameters.  

**Best Parameters:**
```python
{
 'booster': 'gbtree',
 'learning_rate': 0.01,
 'max_depth': 7,
 'min_child_weight': 3,
 'n_estimators': 500,
 'colsample_bytree': 0.6,
 'subsample': 1.0,
 'reg_alpha': 0,
 'reg_lambda': 10.0,
 'random_state': 42
}
