''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets


diabetes = datasets.load_diabetes()


#how many sameples and How many features?
print(diabetes.data.shape)
''' 442 Samples '''
''' 10 Features '''


# What does feature s6 represent?
print(diabetes.DESCR)  #DESCR shows features about the data
''' glu, blood sugar level '''





from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state = 11)

#there are three steps to model something with sklearn

#step 1- setup the model
mymodel = LinearRegression()

#step 2- use fit to train our model
mymodel.fit(x_train, y_train)

#print out the coefficient
print(mymodel.coef_)

#print out the intercept
print(mymodel.intercept_)

#step 3- assign the data
predicted = mymodel.predict(x_test)
expected = y_test

# create a scatterplot with regression line
plt.plot(expected,predicted,".")



#add a line to the graph
x = np.linspace(0,330,100)
#print(x)
y = x
plt.plot(x,y)
plt.show()