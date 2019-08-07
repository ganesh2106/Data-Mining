import urllib.request
from numpy import genfromtxt, zeros
import matplotlib
from tkinter import *
from pylab import plot, show, figure, subplot, hist, xlim, show

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



print (data.shape)
print (target.shape)
#print (set(target))


