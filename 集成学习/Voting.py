import numpy as py
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn import datasets

warnings.filterwarnings('ignore')

iris=datasets.load_iris()
x_data,y_data=iris.data[:,1:3],iris.target

clf1=DecisionTreeClassifier()
clf2=KNeighborsClassifier(n_neighbors=1)
clf3=LogisticRegression()

sclf=VotingClassifier([('dtree',clf1),('knn',clf2),('lr',clf3)])

for clf,lable in zip([clf1,clf2,clf3,sclf],['Decision Tree','KNN','Logistic Regress','Voting']):
    scores=model_selection.cross_val_score(clf,x_data,y_data,scoring='accuracy',cv=3)
    print('Accuracy:{:.2f} [{:s}]'.format(scores.mean(),lable))