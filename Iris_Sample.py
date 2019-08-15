import urllib.request
from numpy import genfromtxt, zeros, mean, linspace, matrix, corrcoef, arange
from numpy.random import rand
import matplotlib
from tkinter import *
from sklearn.naive_bayes import GaussianNB
from pylab import plot, show, figure, subplot, hist, xlim, show, pcolor, colorbar, xticks, yticks
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, classification_report, completeness_score, homogeneity_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


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
print('\n')

# doing unsupervised data analysis using k-means algorithm

print('KMeans method for unsupervised learning\n')
kmeans = KMeans(3, init ='random') #initialization
kmeans.fit(data) #actual execution

c = kmeans.predict(data)

print ("Completeness score:", completeness_score(t,c))
print('\n')
print("Homogeneity score:", homogeneity_score(t,c))



#print (set(target))


# visualize the result of the clustering and compare the assignments with the real labels visually
figure()
subplot(211) #top figure with real classes
plot(data[t==1, 0], data[t==1,2], 'bo')
plot(data[t==2, 0], data [t==2,2], 'ro')
plot(data[t==3, 0], data [t==3,2], 'go')


# subplot(212) # bottom figure with classes assigned automatically
# plot(data[c==1,0],data[t==1,2],'bo',alpha=.7)
# plot(data[c==2,0],data[t==2,2],'go',alpha=.7)
# plot(data[c==0,0],data[t==0,2],'mo',alpha=.7)
# show()

# Regression
print("Regression Analysis")
x = rand(40,1) # explanatory variable
y = x*x*x+rand(40,1)/5 # dependent variable
linreg = LinearRegression()
linreg.fit(x, y)
xx = linspace(0,1,40)
plot(x,y, 'o',xx, linreg.predict(matrix(xx).T), '--r')
show()

# quantify how the model fits the original data using the mean squared error

print(mean_squared_error(linreg.predict(x),y))
print('\n')
print("Correlation\n")
corr = corrcoef(data.T) # .T gives the transpose
print (corr)

pcolor(corr)
colorbar() # add
# arranging the names of the variables on the axis
xticks(arange(0.5,4.5),['sepal length',  'sepal width', 'petal length', 'petal width'],rotation=-20)
yticks(arange(0.5,4.5),['sepal length',  'sepal width', 'petal length', 'petal width'],rotation=-20)
show()

print("\nDimesnsionality Reduction\n")
pca = PCA(n_components=2) #instantiate PCA object
pcad = pca.fit_transform(data)

plot(pcad[target=='setosa',0],pcad[target=='setosa',1],'bo')
plot(pcad[target=='versicolor',0],pcad[target=='versicolor',1],'ro')
plot(pcad[target=='virginica',0],pcad[target=='virginica',1],'go')
show()

# determine how much info stored in PC looking at variance ratio
print (1-sum(pca.explained_variance_ratio_))

data_inv = pca.inverse_transform(pcad) # get original data back
print (abs(sum(sum(data - data_inv))))


for i in range(1,5):
    pca = PCA(n_components=i)
    pca.fit(data)
    print (sum(pca.explained_variance_ratio_) * 100,'%')


