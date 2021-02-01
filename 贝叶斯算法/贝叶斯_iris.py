import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
import warnings
warnings.filterwarnings('ignore')

iris=datasets.load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target)

mul_nb=MultinomialNB()
mul_nb.fit(x_train,y_train)
print(classification_report(y_test,mul_nb.predict(x_test)))
print(confusion_matrix(y_test,mul_nb.predict(x_test)))

print('----------------------------------------------------')

bnl_nb=BernoulliNB()
bnl_nb.fit(x_train,y_train)
print(classification_report(y_test,bnl_nb.predict(x_test)))
print(confusion_matrix(y_test,bnl_nb.predict(x_test)))

print('----------------------------------------------------')

gus_nb=GaussianNB()
gus_nb.fit(x_train,y_train)
print(classification_report(y_test,gus_nb.predict(x_test)))
print(confusion_matrix(y_test,gus_nb.predict(x_test)))