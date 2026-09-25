# Bike Sharing Demand Prediction

A machine learning project that predicts the number of bike rentals per hour using historical rental, weather, and time-related information.

## Project Overview

Bike rental demand changes depending on factors such as time of day, temperature, humidity, weather, season, holidays, and working days.
This project develops a supervised machine learning regression model to predict the total number of bike rentals for a given hour.
The project follows a complete machine learning workflow, including data loading, data inspection, exploratory data analysis, data cleaning, feature engineering, preprocessing, model training, validation, testing, experimentation, model saving, and deployment using Streamlit.

## Objectives

* Understand the structure and characteristics of the bike-sharing dataset.
* Perform exploratory data analysis to identify important patterns.
* Analyze the relationship between bike demand and weather conditions.
* Study how rental demand changes according to time and season.
* Clean and prepare the dataset for machine learning.
* Create useful time-based features.
* Apply suitable preprocessing techniques.
* Train and compare multiple regression models.
* Evaluate models using MAE, RMSE, and R².
* Evaluate the final model on unseen test data.
* Save the trained model and preprocessing steps.
* Develop a working Streamlit application for prediction.

## Dataset

The project uses the Bike Sharing Demand dataset from Kaggle.
The dataset contains 10,886 records and 12 original columns. Each record represents bike rental activity for a particular hour.

### Dataset Features

| Feature                         | Description                                |
| ------------------------------- | ------------------------------------------ |
| 'datetime'                      | Date and time of the rental observation    |
| 'season'                        | Season category                            |
| 'holiday'                       | Indicates whether the day is a holiday     |
| 'workingday'                    | Indicates whether the day is a working day |
| 'weather'                       | Weather condition category                 |
| 'temp'                          | Temperature in Celsius                     |
| 'atemp'                         | Feels-like temperature                     |
| 'humidity'                      | Humidity percentage                        |
| 'windspeed'                     | Windspeed                                  |
| 'casual'                        | Number of rentals made by casual users     |
| 'registered'                    | Number of rentals made by registered users |
| 'count'                         | Total number of bike rentals               |
| The target variable is 'count'. |                                            |

### Target Leakage

The columns 'casual' and 'registered' were removed from the model inputs because:
count = casual + registered
Using these variables as input features would give the model direct information about the target and cause target leakage.

## Exploratory Data Analysis

Several visualizations were created to understand the dataset and identify important patterns.
The analysis included:

* Distribution of bike rental demand
* Temperature vs. rental demand
* Hourly rental demand
* Weather condition vs. rental demand
* Seasonal rental demand
* Working day vs. rental demand
* Humidity vs. rental demand
* Windspeed vs. rental demand
* Correlation heatmap
* Numerical-variable boxplots

### Key EDA Findings

* Rental demand is positively skewed, with many lower-demand observations and fewer periods with very high demand.
* Rental demand changes considerably throughout the day.
* Morning and evening hours show clear demand peaks.
* Rental demand generally increases with temperature up to moderate levels and decreases at very high temperatures.
* Poorer weather conditions are generally associated with lower rental demand.
* Rental demand varies across seasons.
* Working and non-working days have similar overall demand distributions, but the timing of demand differs.
* Higher humidity generally corresponds to lower rental demand.
* Higher windspeed is generally associated with lower rental demand.
* 'temp' and 'atemp' are highly correlated, so 'atemp' was removed from the final feature set.

## Data Cleaning

The dataset was checked for missing values, duplicate records, invalid values, unusual categories, and outliers.

### Missing Values

The original dataset contained no missing values. However, 22 records had humidity recorded as 0. These values were treated as invalid and replaced using the median humidity.
After cleaning, no missing values remained.

### Duplicate Records

The dataset was checked for duplicate rows. No duplicate records were found.

### Weather Categories

Weather category 4 appeared only once. Since it is a valid category, it was retained.

### Outliers

Boxplots showed some high values in 'count' and 'windspeed'. These values were not automatically removed because they may represent genuine observations.

## Feature Engineering

The 'datetime' column was converted into several useful features:

| Feature                                                                                                           | Description                                     |
| ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| 'year'                                                                                                            | Year of the observation                         |
| 'month'                                                                                                           | Month of the observation                        |
| 'day'                                                                                                             | Day of the month                                |
| 'hour'                                                                                                            | Hour of the day                                 |
| 'day_of_week'                                                                                                     | Day of the week                                 |
| 'is_weekend'                                                                                                      | Indicates whether the day is Saturday or Sunday |
| 'hour_sin'                                                                                                        | Cyclic representation of hour                   |
| 'hour_cos'                                                                                                        | Cyclic representation of hour                   |
| The cyclic hour features help the model understand that the end and beginning of the day are close to each other. |                                                 |

## Feature Selection

The following variables were removed:

* 'datetime' because useful information was extracted from it.
* 'casual' and 'registered' because they cause target leakage.
* 'atemp' because it is highly correlated with 'temp'.
  The final feature set contained 15 features:
season
holiday
workingday
weather
temp
humidity
windspeed
hour
year
month
day
day_of_week
is_weekend
hour_sin
hour_cos

## Data Preprocessing

Both numerical and categorical features were present in the dataset.

### Categorical Features

One-Hot Encoding was applied to:
season
holiday
workingday
weather
year
month
day_of_week
is_weekend

### Numerical Features

StandardScaler was applied to:

```text
temp
humidity
windspeed
hour
day
hour_sin
hour_cos
```

A ColumnTransformer was used to apply the appropriate preprocessing to each group of features.
After preprocessing, the 15 input features were converted into 42 model-ready features.

## Train, Validation, and Test Split

Since the dataset contains time-based observations, the data was sorted chronologically before splitting.
A chronological split was used instead of randomly shuffling the data.

| Dataset                                                                                                                                                                      | Records | Percentage |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------: | ---------: |
| Training                                                                                                                                                                     |   7,620 |        70% |
| Validation                                                                                                                                                                   |   1,632 |        15% |
| Testing                                                                                                                                                                      |   1,634 |        15% |
| The training set was used for model training, the validation set was used for model comparison and experimentation, and the test set was kept separate for final evaluation. |         |            |

## Models Used

Three regression approaches were tested.

### Linear Regression

Linear Regression was used as a baseline model.

| Metric | Validation Result |
| ------ | ----------------: |
| MAE    |            134.41 |
| RMSE   |            170.81 |
| R²     |            0.4102 |

### Random Forest Regressor

Random Forest was used to capture non-linear relationships between the features and rental demand.
For the 100-tree model:

| Metric | Validation Result |
| ------ | ----------------: |
| MAE    |             50.50 |
| RMSE   |             77.53 |
| R²     |            0.8785 |

### Gradient Boosting Regressor

Gradient Boosting was also tested.

| Metric | Validation Result |
| ------ | ----------------: |
| MAE    |             69.61 |
| RMSE   |             96.49 |
| R²     |            0.8118 |

## Model Experimentation

Additional Random Forest experiments were performed by changing the number of trees and maximum tree depth.

| Model Configuration                                                                                                                               |   MAE |  RMSE |     R² |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ----: | ----: | -----: |
| 100 trees                                                                                                                                         | 50.50 | 77.53 | 0.8785 |
| 200 trees                                                                                                                                         | 50.70 | 77.62 | 0.8782 |
| Maximum depth 15                                                                                                                                  | 50.61 | 77.68 | 0.8780 |
| Maximum depth 10                                                                                                                                  | 53.62 | 80.29 | 0.8697 |
| Increasing the number of trees from 100 to 200 did not improve the validation results. Reducing the maximum depth to 10 also reduced performance. |       |       |        |

## Model Evaluation

Three metrics were used to evaluate the regression models.

### Mean Absolute Error

MAE measures the average absolute difference between actual and predicted rental counts. A lower value indicates smaller average prediction errors.

### Root Mean Squared Error

RMSE measures prediction error while giving more importance to larger errors. A lower value indicates smaller prediction errors.

### R² Score

R² represents the proportion of variation in rental demand explained by the model. A higher value indicates that the model explains more of the variation in the target.

## Final Model

The final model used in this project is a Random Forest Regressor with 100 trees.
The model was evaluated on the unseen test set.

### Final Test Results

| Metric                                                                                                                            | Result |
| --------------------------------------------------------------------------------------------------------------------------------- | -----: |
| MAE                                                                                                                               |  61.60 |
| RMSE                                                                                                                              |  87.83 |
| R²                                                                                                                                | 0.8322 |
| The R² score of 0.8322 means that the model explains approximately 83.2% of the variation in bike rental demand in the test data. |        |
| The MAE of 61.60 means that the predictions differ from the actual rental counts by about 62 bikes per hour on average.           |        |

## Actual vs. Predicted Analysis

The actual vs. predicted plot shows that the predictions generally follow the overall pattern of the actual rental counts.
The prediction error becomes larger for higher rental counts, and some high-demand periods are underestimated.
This indicates that the model captures the general demand pattern but has more difficulty predicting extreme demand.

## Feature Importance

Feature importance from the Random Forest model showed that the most influential features were:

| Feature                                                                                                                                                                                      | Importance |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------: |
| 'hour'                                                                                                                                                                                       |     0.3837 |
| 'temp'                                                                                                                                                                                       |     0.1478 |
| 'hour_sin'                                                                                                                                                                                   |     0.1444 |
| 'hour_cos'                                                                                                                                                                                   |     0.0746 |
| 'humidity'                                                                                                                                                                                   |     0.0343 |
| 'year_2012'                                                                                                                                                                                  |     0.0278 |
| 'year_2011'                                                                                                                                                                                  |     0.0257 |
| 'workingday_1'                                                                                                                                                                               |     0.0230 |
| The results show that time-related features are important for predicting bike rental demand. In particular, 'hour', 'hour_sin', and 'hour_cos' contribute strongly to the model predictions. |            |

## Overfitting Analysis

The Random Forest model achieved a training R² of approximately 0.992 and a validation R² of approximately 0.879.
This difference indicates some overfitting because the model performs better on the training data than on unseen validation data.
However, the validation and test results show that the model still generalizes reasonably well to unseen observations.

## Model Saving

The preprocessing steps and Random Forest model were combined into a single Scikit-learn Pipeline.
The complete pipeline was saved as:

```text
bike_demand_model.pkl
```

Git LFS was used to store the large model file in the GitHub repository.

## Working Prototype

A Streamlit web application was developed as the working prototype.
The application allows users to enter:

* Date
* Hour of the day
* Temperature
* Humidity
* Windspeed
* Season
* Weather condition
* Holiday status
* Working-day status
  After the user clicks the prediction button, the application displays the predicted number of bike rentals per hour.

## Project Structure

```text
Bike-Demand-Prediction/
│
├── bike_rental.ipynb
├── app.py
├── bike_demand_model.pkl
├── requirements.txt
├── .gitattributes
└── README.md
```

### Files

'bike_rental.ipynb' contains the complete machine learning workflow, including data loading, EDA, cleaning, feature engineering, preprocessing, model training, experimentation, evaluation, and model saving.
'app.py' contains the Streamlit working prototype.
'bike_demand_model.pkl' contains the saved preprocessing pipeline and trained Random Forest model.
'requirements.txt' contains the required Python packages and versions.
'.gitattributes' configures Git LFS for the model file.
'README.md' contains the project documentation.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Git
* GitHub
* Git LFS

## How to Run the Project

### Clone the Repository

```bash
git clone https://github.com/rashhmii18/Bike-Demand-Prediction.git
cd Bike-Demand-Prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in a web browser.

## Key Findings

* Bike rental demand changes significantly throughout the day.
* Morning and evening periods show clear demand peaks.
* Temperature has a non-linear relationship with rental demand.
* Poorer weather conditions are generally associated with lower rental demand.
* Higher humidity tends to be associated with lower rental demand.
* Working days and non-working days have similar overall demand distributions, but the timing of demand differs.
* 'hour' is the most influential feature in the final Random Forest model.
* 'hour_sin' and 'hour_cos' help represent the daily cycle.
* The final model achieved an R² of 0.8322 on the unseen test set.
* The model has more difficulty predicting extreme high-demand periods.

## Conclusion

This project successfully developed a machine learning model for predicting hourly bike rental demand.
The analysis showed that time of day, temperature, weather, humidity, and calendar-related features are useful for understanding rental demand. Feature engineering and preprocessing were applied to prepare the data for machine learning.
Several regression models were tested, including Linear Regression, Random Forest, and Gradient Boosting. The final Random Forest model achieved an MAE of 61.60, an RMSE of 87.83, and an R² of 0.8322 on the unseen test set.
The model was saved as a reusable pipeline and integrated into a Streamlit application, making it possible to enter new information and receive a bike rental demand prediction.
Overall, the project demonstrates a complete machine learning workflow from data analysis and preprocessing to model development, evaluation, model saving, and deployment.

## Dataset Source

Kaggle Bike Sharing Demand dataset.

## Project

Bike Sharing Demand Prediction
Machine Learning Regression Project
Python | Scikit-learn | Random Forest | Streamlit | GitHub
