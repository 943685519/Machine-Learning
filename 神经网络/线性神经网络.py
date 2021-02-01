import numpy as np
import matplotlib.pyplot as plt

X_data=np.array([[1,3,3],
                 [1,4,3],
                 [1,1,1],
                 [1,0,2]])
Y_data=np.array([[1],
                 [1],
                 [-1],
                 [-1]])
W=(np.random.random([3,1])-0.5)*2       #random自动生成一个3行一列的，取值在0~1之间的矩阵
print(W)

lr=0.11
#神经网络的输出
O=0


def updata():
    global X_data,Y_data,W,lr
    O=np.dot(X_data,W)
    W_C= lr*(X_data.T.dot(Y_data-O))/int(X_data.shape[0])
    W=W+W_C

for i in range(100):
    updata()

x1=[3,4]
y1=[3,3]

x2=[1,0]
y2=[1,2]

k=-W[1]/W[2]
d=-W[0]/W[2]
print('k=',k)
print('d=',d)

xdata=np.array([ [0],[5] ])
print(xdata.shape)


plt.figure()
plt.plot(xdata,xdata*k+d,'r')
plt.scatter(x1,y1,c='b')
plt.scatter(x2,y2,c='y')
plt.show()