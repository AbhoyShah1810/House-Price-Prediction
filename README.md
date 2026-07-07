# House Price Prediction Notebook

This README describes the `main.ipynb` notebook in `House-Price-Prediction/ML_Model/src/`.
It explains the dataset, the preprocessing steps, the methods used, and the evaluation of the models.
This document is written so you can clearly explain the project to your professor.

---

## Project Overview

The notebook builds a regression pipeline to predict house sale prices using the Ames Housing dataset.
It covers data loading, feature exploration, missing value handling, categorical encoding, train-test splitting, model training, and model comparison.

The key models used are:
- Linear Regression
- Decision Tree Regression

The evaluation metrics are:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

---

## Dataset Files

The notebook uses these files:
- `../data_set/train.csv` — training dataset with house features and the target variable `SalePrice`
- `../data_set/test.csv` — separate test dataset (loaded for completeness, not used for final prediction evaluation in the notebook)
- `../data_set/data_description.txt` — feature definitions and category meanings for the Ames Housing dataset

---

## Target Variable

- `SalePrice`: The final sale price of the house in dollars. This is the target variable the models predict.

---

## Dataset Features

The Ames Housing dataset includes the following columns. Each feature is listed with a short explanation and any coded values.

### General and Location Features

- `Id` — Unique identifier for each row.
- `MSSubClass` — Dwelling type; numeric code representing style and construction era.
- `MSZoning` — General zoning classification for the sale.
- `LotFrontage` — Linear feet of street connected to the property.
- `LotArea` — Lot size in square feet.
- `Street` — Type of road access.
- `Alley` — Type of alley access.
- `LotShape` — General shape of the property.
- `LandContour` — Flatness of the property.
- `Utilities` — Types of utilities available.
- `LotConfig` — Lot configuration.
- `LandSlope` — Slope of the property.
- `Neighborhood` — Physical location within the city of Ames.
- `Condition1` — Proximity to major street or railroad.
- `Condition2` — Secondary proximity condition if more than one applies.

### House Structure and Style

- `BldgType` — Type of dwelling.
- `HouseStyle` — Style of dwelling.
- `OverallQual` — Overall material and finish quality.
- `OverallCond` — Overall condition rating.
- `YearBuilt` — Original construction year.
- `YearRemodAdd` — Remodel year or addition year.
- `RoofStyle` — Roof style.
- `RoofMatl` — Roof material.
- `Exterior1st` — Exterior material covering first layer.
- `Exterior2nd` — Exterior material covering second layer.
- `MasVnrType` — Masonry veneer type.
- `MasVnrArea` — Masonry veneer area in square feet.
- `ExterQual` — Exterior material quality.
- `ExterCond` — Exterior material condition.
- `Foundation` — Foundation type.

### Basement Features

- `BsmtQual` — Basement height quality.
- `BsmtCond` — Basement condition.
- `BsmtExposure` — Basement exposure to sunlight or walkout.
- `BsmtFinType1` — Rating of basement finished area type 1.
- `BsmtFinSF1` — Type 1 finished square feet.
- `BsmtFinType2` — Rating of basement finished area type 2.
- `BsmtFinSF2` — Type 2 finished square feet.
- `BsmtUnfSF` — Unfinished basement square feet.
- `TotalBsmtSF` — Total basement square feet.

### Heating and Living Area

- `Heating` — Type of heating system.
- `HeatingQC` — Heating quality and condition.
- `CentralAir` — Central air conditioning yes/no.
- `Electrical` — Electrical system type.
- `1stFlrSF` — First floor square feet.
- `2ndFlrSF` — Second floor square feet.
- `LowQualFinSF` — Low quality finished square feet.
- `GrLivArea` — Above grade living area square feet.
- `BsmtFullBath` — Basement full bathrooms count.
- `BsmtHalfBath` — Basement half bathrooms count.
- `FullBath` — Full bathrooms above grade count.
- `HalfBath` — Half bathrooms above grade count.
- `BedroomAbvGr` — Bedrooms above grade count.
- `KitchenAbvGr` — Kitchens above grade count.
- `KitchenQual` — Kitchen quality rating.
- `TotRmsAbvGrd` — Total rooms above grade (excluding bathrooms).
- `Functional` — Home functionality; typical or with deductions.

### Fireplace, Garage, and Exterior Features

- `Fireplaces` — Number of fireplaces.
- `FireplaceQu` — Fireplace quality.
- `GarageType` — Garage location/type.
- `GarageYrBlt` — Year garage was built.
- `GarageFinish` — Garage interior finish.
- `GarageCars` — Garage car capacity.
- `GarageArea` — Garage area in square feet.
- `GarageQual` — Garage quality.
- `GarageCond` — Garage condition.
- `PavedDrive` — Paved driveway quality.
- `WoodDeckSF` — Wood deck area in square feet.
- `OpenPorchSF` — Open porch area in square feet.
- `EnclosedPorch` — Enclosed porch area in square feet.
- `3SsnPorch` — Three-season porch area in square feet.
- `ScreenPorch` — Screen porch area in square feet.
- `PoolArea` — Pool area in square feet.
- `PoolQC` — Pool quality.
- `Fence` — Fence quality.
- `MiscFeature` — Miscellaneous feature not covered elsewhere.
- `MiscVal` — Value of miscellaneous feature.

### Sale and Timing Features

- `MoSold` — Month sold.
- `YrSold` — Year sold.
- `SaleType` — Type of sale.
- `SaleCondition` — Condition of sale.

---

## Notebook Workflow explained

### 1. Import Libraries

The notebook imports libraries for:
- `numpy`, `pandas` for data manipulation
- `sklearn` for model building and evaluation
- `CategoricalDtype` to preserve ordered categories before encoding

### 2. Load the Dataset

The notebook reads the Ames Housing training and test CSV files.
It prints the shape of both datasets so you can see how many records and features exist.

### 3. Quick Data Inspection

It shows:
- the first few rows of training data
- data types and missing value counts
- summary statistics for numerical features
- the distribution of `SalePrice`

This step helps in understanding what features are available and which need cleaning.

### 4. Feature Exploration

The code separates numeric features from categorical ones:
- numeric features are useful for statistical summaries
- categorical features need encoding before model training

### 5. Missing Value Analysis

The notebook counts missing values and sorts the features with missing data.
This is important because the Ames dataset contains many missing values, and each missing value may have different meaning depending on the feature.

### 6. Handle Missing Values

The notebook uses specific fill strategies:
- Features where missing means "not present" are filled with `NA`.
- Categorical missing values are filled with the most frequent value (mode).
- Basement and garage numeric features are filled with `0` when absence means the feature does not exist.
- Remaining numeric features are filled with the median.

This approach is chosen to preserve information while avoiding dropping data.

### 7. Encode Categorical Features

Encoding is performed in two parts:
- ordinal encoding for ordered categories such as `ExterQual`, `KitchenQual`, `FireplaceQu`, `GarageQual`, and others.
- one-hot encoding for the remaining categorical text fields using `pd.get_dummies`.

Ordinal encoding preserves rank order where quality ratings matter. One-hot encoding converts nominal categories into binary columns usable by regression models.

### 8. Prepare Features and Target

The notebook separates:
- `X` — feature matrix with all predictors
- `y` — target variable `SalePrice`

It drops the `Id` field because it is only an identifier and not predictive.

### 9. Train-Test Split

The data is split into training and testing sets using an 80/20 ratio.
This is essential to evaluate model performance on data the model did not see during training.

### 10. Feature Scaling

For Linear Regression, the notebook applies `StandardScaler` so that features have mean 0 and standard deviation 1.
This improves convergence and makes features comparable.

### 11. Model Training

Two models are trained:
- `LinearRegression` with scaled features.
- `DecisionTreeRegressor` with unscaled features.

The Decision Tree model is not scaled because tree-based models do not require feature scaling.

### 12. Prediction and Comparison

The notebook evaluates both models using:
- MAE — average absolute difference between predictions and actual values
- RMSE — square-root of average squared difference, which penalizes larger errors
- R² — proportion of variance explained by the model

A comparison table and summary are printed.

---

## Why this approach is good

- The project covers both data preparation and predictive modeling.
- It handles missing values carefully rather than dropping rows.
- It distinguishes between ordinal and nominal categorical features.
- It compares a simple linear model with a tree-based model.
- It uses clear evaluation metrics to justify which model performs better.

---

## Key points to explain to your professor

- `SalePrice` is the prediction target.
- `Id` is dropped because it is not an input feature.
- Missing values are handled differently depending on context.
- Ordinal categorical variables keep their rank order.
- One-hot encoding is used for nominal categories.
- Linear Regression needs feature scaling; Decision Tree does not.
- MAE, RMSE, and R² give different perspectives on model quality.

---

## Possible improvements

If asked what could be improved, mention:
- cross-validation to make evaluation more robust
- hyperparameter tuning for the Decision Tree and Linear Regression
- using ensemble models like Random Forest or XGBoost
- feature selection or feature engineering to reduce noise
- using the `test.csv` file for final submission-style predictions
- adding visualization of prediction errors and residuals

---

## Suggested answers for professor questions

- **Why use both Linear Regression and Decision Tree?**
  - Linear Regression is a simple baseline model that captures linear relationships.
  - Decision Trees capture non-linear patterns and interactions.

- **Why does the notebook use median for some missing values?**
  - The median is robust to outliers and preserves central tendency without being skewed by extreme values.

- **Why are some numeric columns converted to strings before encoding?**
  - Columns like `MSSubClass`, `YearBuilt`, and `YrSold` are numeric codes or labels, not continuous measurements, so they are treated as categories.

- **Why drop the first dummy column in one-hot encoding?**
  - Dropping one dummy column avoids multicollinearity in linear models.

- **Why compare MAE, RMSE, and R² together?**
  - MAE measures average error directly.
  - RMSE penalizes larger mistakes more strongly.
  - R² shows how much variance the model explains.

---

## File location and usage

Open `main.ipynb` in `House-Price-Prediction/ML_Model/src/`.
Run the notebook cells in order to reproduce the full analysis from data loading through model comparison.

If you want, I can also add a shorter summary slide-style version for exam preparation.
