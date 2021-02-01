import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

x_data=np.r_[np.random.randn(50,2)-[2,2],np.random.randn(50,2)+[2,2]]
y_data=[0]*50+[1]*50
plt.scatter(x_data[:,0],x_data[:,1],c=y_data)

model=svm.SVC(kernel='linear')
model.fit(x_data,y_data)
print('系数和截距：\n',model.intercept_,model.coef_)

k=-model.coef_[0][0]/model.coef_[0][1]
b=model.intercept_[0]/model.coef_[0][1]
b1=model.support_vectors_[0][1]-k*model.support_vectors_[0][0]
b2=model.support_vectors_[1][1]-k*model.support_vectors_[1][0]

x_test=np.array([[-5],[5]])

y_test=k*x_test+b
y_test1=k*x_test+b1
y_test2=k*x_test+b2
print('y_test: \n',y_test)

print("支持向量：\n",model.support_vectors_)

plt.plot(x_test,y_test,c='k')
plt.plot(x_test,y_test1,'r--')
plt.plot(x_test,y_test2,'b--')
plt.show()