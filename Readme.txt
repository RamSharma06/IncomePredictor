Adult Income Prediction Project
Overview
This project entails predicting adult income levels using machine learning techniques in Python. The provided Python script performs data preprocessing, exploratory data analysis (EDA), feature engineering, and implements a RandomForestClassifier to predict income levels based on demographic and socio-economic features.

Requirements
Python 3.x
Pandas
NumPy
Seaborn
Matplotlib
Scikit-learn
Usage
Data Loading and Exploration:

The script reads the 'adult.csv' dataset using Pandas.
Conducts initial exploration using value_counts() for 'occupation' and 'workclass' columns.
One-Hot Encoding and Feature Engineering:

Utilizes pd.get_dummies() to create binary features for categorical columns ('occupation', 'workclass', 'relationship', 'native-country', 'marital-status', 'race').
Encodes 'gender' and 'income' columns as binary features.
Correlation Analysis:

Calculates correlations between features and the target variable ('income').
Visualizes feature correlations using a heatmap via Seaborn and Matplotlib.
Feature Importance and Model Training:

Drops less correlated features to 'income'.
Prepares data for training and testing.
Initializes and trains a RandomForestClassifier for income prediction.
Evaluates model accuracy on the test set.
Feature Importance Calculation:

Extracts feature importances from the trained RandomForestClassifier.
Running the Code
Ensure Python and required libraries are installed.
Download the 'adult.csv' dataset or specify the correct file path accordingly.
Run the Python script in an environment supporting the listed dependencies.