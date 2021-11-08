import pandas as pd
from sklearn.model_selection import train_test_split

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

print(nyc.head(3)) #returns head row
print(nyc.Date.values) #returns a numpy array containing the Date column's values
print(nyc.Date.values.reshape(-1,1)) #reshape tells to infer the number of rows based on number of columns (1) and number of elements (124)

x_train, x_test, y_train, y_test = train_test_split(nyc.Date.values.reshape(-1,1),nyc.Temperature.values) #brings in the 2nd column (temps) as our target

#create the linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X=x_train, y=y_train)
coef = lr.coef_ #m of the slope equation
intercept = lr.intercept_ #b of the slope equation

# testing the model
predicted = lr.predict(x_test)
expected = y_test


#print(predicted[:20])
#print(expected[:20])

predict = lambda x: coef * x + intercept

print(predict(2025))

#visualize it
import seaborn as sns

axes = sns.scatterplot(data=nyc,x="Date",y="Temperature",
                        hue="Temperature", palette="winter",legend=False)

axes.set_ylim(10,70)

import numpy as np

x = np.array([min(nyc.Date.values),max(nyc.Date.values)])
print(x)
y = predict(x)
print(y)

import matplotlib.pyplot as plt

line = plt.plot(x,y)

plt.show()
