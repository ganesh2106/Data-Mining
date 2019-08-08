import urllib.request
from numpy import genfromtxt, zeros, mean
import matplotlib
from tkinter import *
from sklearn.naive_bayes import GaussianNB
from pylab import plot, show, figure, subplot, hist, xlim, show
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score


# target url to download data set
url =  'http://aima.cs.berkeley.edu/data/iris.csv'
result = urllib.request.urlopen(url)
resulttext = result.read()

# writing the data to an external csv file
localFile = open('iris.csv', 'wb')
localFile.write(resulttext)
localFile.close()

# read the first 4 columns
data = genfromtxt('iris.csv', delimiter=',', usecols =(0,1,2,3))

# read the fifth column
target = genfromtxt('iris.csv', delimiter =',', usecols=(4), dtype = str)


# set(['setosa', 'versicolor', 'virginica'])

# plotting the values of a feature against the values of another one
# in this case plotting the sepal length and sepal width of each plant

plot(data[target =='setosa', 0], data[target=='setosa', 2], 'bo')
plot(data[target =='versicolor', 0], data[target=='versicolor', 2], 'ro')
plot(data[target =='virginica', 0], data[target=='virginica', 2], 'go')
show()

# plot the distribution of the first feature of our data (sepal length) for each class

xmin = min(data[:, 0])
xmax = max(data[:, 0])
figure()

subplot(411) # distribution of the setosa class
hist(data[target=='setosa', 0], color = 'b', alpha=.7)
xlim(xmin, xmax)

subplot(412) # distribution of versilcolor class
hist(data[target=='versicolor', 0], color = 'r', alpha=.7)
xlim(xmin, xmax)

subplot(413) # distribution of virginica class
hist(data[target=='virginica', 0], color = 'g', alpha=.7)
xlim(xmin, xmax)

subplot(414) # global histogram (4th, on the bottom)
hist(data[:,0],color='y',alpha=.7)
xlim(xmin,xmax)

show()

# use Gaussian Naive Bayes from sklearn to identify iris flowers
# converting each class into integers

t = zeros(len(target))
t[target == 'setosa'] = 1
t[target == 'versicolor'] = 2
t[target == 'virginica'] = 3

#instantiate and train the classifier

classifier = GaussianNB()
classifier.fit(data, t ) # training on the iris dataset

print (classifier.predict(data[[0]]))

train, test, t_train, t_test = model_selection.train_test_split(data, t, test_size = 0.4, random_state=0)

# train
classifier.fit(train, t_train)

# test
print (classifier.score(test, t_test))

# testing performance of classifier using a confusion matrix
print (confusion_matrix(classifier.predict(test), t_test))

# print a complete report in the performance of the classifier

print (classification_report(classifier.predict(test), t_test, target_names=['setosa', 'versicolor', 'virginica']))

# evaluate the classifier and compare it with others using Cross Validation
# cross validation with 6 iterations

scores = cross_val_score(classifier, data, t, cv =6)
print(scores)
print(mean(scores))

#print (set(target))


