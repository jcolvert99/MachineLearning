# The Iris dataset is referred to as a “toy dataset” because it has only 150 samples and four features. 
# The dataset describes 50 samples for each of three Iris flower species—Iris setosa, Iris versicolor and Iris 
# virginica. Each sample’s features are the sepal length, sepal width, petal 
# length and petal width, all measured in centimeters. The sepals are the larger outer parts of each flower 
# that protect the smaller inside petals before the flower buds bloom.

#EXERCISE
# load the iris dataset and use classification
# to see if the expected and predicted species
# match up
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()

# display the shape of the data, target and target_names
print(iris.data.shape)
print(iris.target.shape)
print(iris.target_names)  #flower names are indexed at 0, 1, and 2


# display the first 10 predicted and expected results using
# the species names not the number (using target_names)
data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, random_state=11) 
 
'''
what the train, test, split does: took original data set with 150 rows and split it up into the 4 datasets (112 rows for training, 28 for testing)
    -uses 75% for training the model and 25% for testing the model
    -have to split it up because one dataset, however if the assignment gave you one train and one test dataset, then you wouldn't
     need to split it up

print(data_train.shape)
print(target_train.shape)
print(data_test.shape)
print(target_test.shape)
'''

from sklearn.neighbors import (KNeighborsClassifier)
knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train)  
predicted = knn.predict(X=data_test)   
expected = target_test

print(predicted[:10])
print(expected[:10])
#assingment wants target names not class number         
print(iris.target_names)
predicted = [iris.target_names[x] for x in predicted]
expected = [iris.target_names[x] for x in expected]

print(predicted[:10])
print(expected[:10])


# display the values that the model got wrong
wrong = [(p,e) for (p,e) in zip(predicted,expected) if p != e]
print(wrong)


# visualize the data using the confusion matrix
from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)
confusion_df = pd.DataFrame(confusion, index=iris.target_names, columns=iris.target_names)

figure = plt.figure()
axes = sns.heatmap(confusion_df, annot=True, cmap=plt.cm.nipy_spectral_r)
plt.xlabel("Expected")
plt.ylabel("Predicted")

plt.show()
