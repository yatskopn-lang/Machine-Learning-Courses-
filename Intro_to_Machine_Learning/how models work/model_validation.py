"""Model Validation"""

# Code I have previously used to load data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = [
    'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
    'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'
    ]
X = home_data[feature_columns]
# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)
print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex4 import *
print("Setup Complete")

#Step 1: Split Your Data
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

#Step 2: Specify and Fit the Model
# Specify the model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)

#Step 3: Make Predictions with Validation data
val_predictions = iowa_model.predict(val_X)
print(iowa_model.predict(val_X.head()))
print(val_y.head())

#Step 4: Calculate the Mean Absolute Error in Validation Data
from sklearn.metrics import mean_absolute_error
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)

