# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Importing Dataset 

dataset = pd.read_csv('Data.csv')
# print(dataset)

x = dataset.iloc[:, :-1].values
# print(x)

y = dataset.iloc[:, -1].values
# print(y)


# Handeling Missing Datas

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3]) 
x[:, 1:3] = imputer.transform(x[:, 1:3])

# print(x)


# Encoding Categorical Data
# for independent variable (OneHotEncoder())
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)


#for Dependent variable (LabelEncoder())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)
print(y)


# Split into Train and Test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train)
print(x_test)
print(y_train)
print(y_test) 


# Feature Scaling 

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])

# not fit_transform in x_test
x_test[:, 3:] = sc.transform(x_test[:, 3:]) 

print(x_train)
print(x_test)


