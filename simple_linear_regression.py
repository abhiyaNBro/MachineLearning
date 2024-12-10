import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv("Salary_Data.csv")
print(dataset)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


from sklearn.model_selection import train_test_split

x_train , x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 0)

print(x_train)
print(x_test)
print(y_train)
print(y_test)


# TRAINING SIMPLE LINEAR regression model on the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)



#  Predicting the test set result

y_pred = regressor.predict(x_test)


# Visualization for training set

plt.scatter(x_train, y_train, color= 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("Salary vs Years Experience (training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()



# Visualization for Test set

plt.scatter(x_test, y_test, color= 'red')
plt.plot(x_test, regressor.predict(x_test), color = 'blue')
plt.title("Salary vs Years Experience (test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()