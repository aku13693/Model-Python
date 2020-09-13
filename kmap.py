from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
#loading datasets
iris = datasets.load_iris()

#Printing description and features
features = iris.data
label = iris.target
print(iris.DESCR)
print(features[0] , label[0])

#Training the classifier

clf = KNeighborsClassifier()
clf.fit(features,label)

#Predicting

print(clf.predict([[1,2,3,4]]))