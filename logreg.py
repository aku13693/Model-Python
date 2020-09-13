
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import  numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()

#print(list(iris.keys()))

#print(iris.data)
#print(iris.target)

#print(iris.data.shape)

X = iris.data[:, 3:]

y = (iris.target == 2).astype(np.int)

clf = LogisticRegression()
clf.fit(X,y)
c = clf.predict(([[0]]))
print(c)

anew = np.linspace(0,3,1000).reshape(-1,1)
#print(anew)

aproba = clf.predict_proba(anew)
print(aproba[0])
plt.plot(anew,aproba[:,1])
plt.show()