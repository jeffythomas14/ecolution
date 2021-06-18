import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn import LinearRegression
import joblib

import json

sns.set()

data = pd.read_csv('CarbonFootprint.csv')
data.head()

Y = data['Footprint']
X = data[['Age','Location','Occupation']]

X = pd.get_dummies(data=X, drop_first=False)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=101)



model = LinearRegression()
model.fit(X_train,y_train)


print("hello")