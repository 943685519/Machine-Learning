import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import warnings

warnings.filterwarnings('ignore')

data=np.genfromtxt('LR-testSet2.txt',delimiter=',')
x_data=data[:,:2]
y_data=data[:,2]
x0_data=[]
y0_data=[]
x1_data=[]
y1_data=[]

for i in range(len(x_data)):
    if y_data[i]==0:
        x0_data.append(x_data[i][0])
        y0_data.append(x_data[i][1])
    else:
        x1_data.append(x_data[i][0])
        y1_data.append(x_data[i][1])



model=svm.SVC(kernel='rbf')
model.fit(x_data,y_data)
print(model.score(x_data,y_data))

x_min,x_max=x_data[:,0].min()-1,x_data[:,0].max()+1
y_min,y_max=x_data[:,1].min()-1,x_data[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),
                  np.arange(y_min,y_max,0.02))
z=model.predict(np.c_[xx.ravel(),yy.ravel()])
z=z.reshape(xx.shape)
cs=plt.contourf(xx,yy,z)

scatter0=plt.scatter(x0_data,y0_data,c='r',marker='x')
scatter1=plt.scatter(x1_data,y1_data,c='b',marker='o')
plt.legend(handles=[scatter0,scatter1],labels=['kind0','kind1'],loc='best')
plt.show()