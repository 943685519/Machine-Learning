from sklearn import datasets
import numpy as np
from sklearn import model_selection
from mlxtend.classifier import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

iris=datasets.load_iris()
x_data,y_data=iris.data[:,1:3],iris.target

#3个分类器
clf1=KNeighborsClassifier(n_neighbors=1)
clf2=DecisionTreeClassifier()
clf3=LogisticRegression()

#1个次级分类器
lr=LogisticRegression()
sclf=StackingClassifier(classifiers=[clf1,clf2,clf3],meta_classifier=lr)

for clf,label in zip([clf1,clf2,clf3,sclf],['KNN','Decision Tree','LogisticRegression','StackingClassifier']):
    scores=model_selection.cross_val_score(clf,x_data,y_data,cv=3,scoring='accuracy')   #交叉验证后的值，模型是clf
    print('Accuracy:%0.2f [%s]' % (scores.mean(), label))                               #数据是x_data,y_data
                                                                                        #cv是数据分为3部分，两部分为训练，一部分测试
                                                                                        #scoring是要计算的，accuracy是准确率
