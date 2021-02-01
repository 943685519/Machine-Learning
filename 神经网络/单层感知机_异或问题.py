import numpy as np
import matplotlib.pyplot as plt

X=np.array([[1,0,0],
            [1,0,1],
            [1,1,0],
            [1,1,1]])
Y=np.array([[-1],
            [1],
            [1],
            [-1]])
W=(np.random.random([3,1])-0.5)*2
print(W)
lr=0.01
O=0

def updata():
    global X,Y,W,lr
    O=np.sign(np.dot(X,W))
    O=np.mat(O)
    W_C= lr*(X.T.dot(Y-O))/int(X.shape[0])
    W=W+W_C

for i in range(100):
    updata()
    print(W)
    print(i)
    O=np.sign(np.dot(X,W))
    if(O==Y).all():
        print('Finished')
        print('epoch:',i)
        break

k=-W[1]/W[2]
d=-W[0]/W[2]
print('k=',k)
print('d=',d)

x1=[0,1]
y1=[1,0]
x2=[0,1]
y2=[0,1]

xtest=np.array([[-1],[3]])

plt.figure()
plt.plot(xtest,xtest*k+d,'r')
plt.scatter(x1,y1,c='b')
plt.scatter(x2,y2,c='y')
plt.show()

#结论：单层感知机没办法做异或问题