import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_salaries.csv")

print(dataset)

x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(x, y)

# Training Linear regression on whole dataset

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x, y)


# Training Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


# Visualizing the Linear Regression Result


plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show()


# Visualizing the Polynomial Regression Result

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(x_poly), color = 'blue')
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show()


# Predicting the New result with Linear Regression

print(lin_reg.predict([[6.5]]))

# Predicting the New result with Polynomial Regression

print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))