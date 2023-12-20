# Problem statement

### A data set containing real estate information is available in different zip codes in New York City. The data set includes details such as zip code, property price, number of bedrooms, number of bathrooms, living space, address, city, state, zip code population, zip code density, county, median household income, latitude and longitude.

### The goal is to develop a predictive model that can estimate the price of a property based on its characteristics, such as the number of bedrooms, number of bathrooms, living space, etc. This model will be useful in providing buyers and sellers with an accurate estimate of a property's value based on specific factors.



# Dataset 
### "https://www.kaggle.com/datasets/jeremylarcher/american-house-prices-and-demographics-of-top-cities/"

# Steps to solve

## Exploratory Data Analysis (EDA):
### Perform an EDA to understand the distribution of each variable, identify outliers, and evaluate the correlation between characteristics and property price.

## Data Cleaning:
### Identify and handle duplicates, missing or inconsistent values ​​in the data set.

## Feature Engineering:
### Create new relevant features to improve the predictive ability of the model.

## Division of the Data Set:
### Separate the data set into training and test sets to evaluate the performance of the model.

## Modeling:
### Select an appropriate regression model to predict the property price. Train the model using the training set and tune the hyperparameters to optimize performance.

## Model Validation:
### Evaluate the model using the test set and evaluation metrics such as mean square error (MSE) or coefficient of determination (R²).

## Interpretation of results:
### Analyze the importance of the features and how they affect the price of the property.

## Deployment:
### Deploy the trained model to a production environment so that it can be used by stakeholders.

# Project Description

### In this project, we explored various machine learning models to predict math scores. The following models were trained and evaluated:

- GradientBoostingClassifier
- RandomForestClassifier
- DecisionTreeClassifier
- LinearRegression
- LogisticRegression

# Repository Structure

- **README.md: The current file.
- **capstone1.ipynb: Jupyter Notebook containing data preparation, EDA, and model selection.
- **requirements.txt: Text file listing all the required packages.
- **app.py: Python script for the future deployment of the model using Flask.

## Getting Started

To replicate the environment used in this project, create a virtual environment using the following commands:

```bash
conda create -p venv python==3.8 -y
conda activate venv
pip install -r requirements.txt