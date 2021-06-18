import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression
import joblib

import json

sns.set()

data = pd.read_csv('CarbonFootprint.csv')


include= ['Age','Location','Occupation','Footprint']
Y = data[include]

df = pd.get_dummies(Y, drop_first=False)

dependent_variable= 'Footprint'
x= df[df.columns.difference([dependent_variable])]
y= df[dependent_variable]

model = LinearRegression()
model.fit(x,y)


joblib.dump(model, 'model1.pkl')
print("Model dumped!")


# Load the model that you just saved
model = joblib.load('model1.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns1.pkl')
print("Models columns dumped!")