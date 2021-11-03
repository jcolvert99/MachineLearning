from sklearn.datasets import load_digits

digits = load_digits()


#print(digits.DESCR)

#print(digits.data[5])  #each number reveals how dark the pixel is (creates the image of 5 with the sample for 5)


#print(digits.target[5])
#target is a number between 0 and 9 - represents what the digit is in terms of shapes- repeats the pixel numbers
#5 is one sample (row) of the 1797


#print(digits.data.shape)  #returns 1797 rows (samples) with 64 columns (features) that are translated into a set of numbers that the machine reads as output
#print(digits.target.shape)  #target has only 1 column because each row represents a number, and the target attribute is that number
                            #target is only one answer to what that data is

import matplotlib.pyplot as plt
figure, axes = plt.subplots(nrows=4,ncols=6,figsize=(6,4))

for item in zip(axes.ravel(),digits.images,digits.target):
    axes,image,target = item
    axes.imshow(image,cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)

plt.tight_layout()

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    digits.data, digits.target, random_state=11
) #make sure data is split the correct way

print(data_train.shape) #2-dimensional
print(data_test.shape) #gonna test it with another 450 rows and have it give us an answer (predictive)
print(target_train.shape) #1-dimensional
print(target_test.shape)

from sklearn.neighbors import (KNeighborsClassifier)

knn = KNeighborsClassifier()

knn.fit(X=data_train, y=target_train)  # fit is the method that is performing all the machine learning
                                       # it needs both the data and the target to tell it what that data represents
                                       # these numbers represent the number 0, these represent.... (target allows it to do the learning)

predicted = knn.predict(X=data_test)    #don't need target y because it is learning, predicting, and spitting out the answer
expected = target_test

print(predicted[:20])
print(expected[:20])

print(format(knn.score(data_test,target_test),".2%"))

wrong = [(p,e) for (p,e) in zip(predicted,expected) if p != e]

print(wrong)

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_true=expected, y_pred=predicted)
print(confusion)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))

figure = plt2.figure(figsize=(7,6))
axes = sns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral_r)

plt2.show()