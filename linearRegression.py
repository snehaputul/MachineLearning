import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1:2].values

# visualizing the data
plt.scatter(X, y)

# Creating a regressor object
regressor = linear_model.LinearRegression()
regressor.fit(X, y)

y_pred = regressor.predict(X)
plt.scatter(X, y)
plt.plot(X, y_pred)
print(regressor.coef_)
print(regressor.intercept_)
 