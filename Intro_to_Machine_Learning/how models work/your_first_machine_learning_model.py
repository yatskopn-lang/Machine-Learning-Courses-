"""Your First Machine Learning Model"""

# Code I have previously used to load data
import pandas as pd
# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *
print("Setup Complete")

#Step 1: Specify Prediction Target
# print the list of columns in the dataset to find
# the name of the prediction target
y = home_data.columns
print(y)
y = home_data.SalePrice

#Step 2: Create X
# Create the list of features below
feature_names = [
    'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
    'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'
    ]
# Select data corresponding to features in feature_names
X = home_data[feature_names]

#Review Data
print(X)
print(y)

#Step 3: Specify and Fit Model
from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit the model
iowa_model.fit(X, y)

#Step 4: Make Predictions
predictions = iowa_model.predict(X)
print(predictions)

#Think About Your Results
print(y.head() == iowa_model.predict(X.head()))
