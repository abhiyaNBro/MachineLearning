import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')

print(dataset)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# One hot encoding 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(), [3])], remainder='passthrough')

x = np.array(ct.fit_transform(x))


from sklearn.model_selection import train_test_split

x_train , x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

print(x_train , x_test, y_train, y_test)


# training Multiple Linear Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)



# predicting result:

y_pred = regressor.predict(x_test)
np.set_printoptions(precision =2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))




