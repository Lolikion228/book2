import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
class Perceptron(object):
    def __init__(self,lr=0.01,n_iter=50,random_state=1,shuffle=True):
        self.lr=lr
        self.n_iter=n_iter
        self.random_state = random_state
        self.rgen = np.random.RandomState(self.random_state)
        self.w_initialized = False
        self.shuffle=shuffle
    def fit(self,X,y):
        self.cost_ = []
        self._init_weights(X.shape[1])
        for n in range(n_iter):
            if self.shuffle: X,y=self._shufle(X,y)
            cost=[]
            for xi,target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            self.cost_.append(sum(cost)/len(cost))
        return self

    def partial_fit(self,X,y):
        if not self.w_initialized:
            self._init_weights(X.shape[1])
        for xi,target in zip(X,y):
            self._update_weights(xi,target)
        return self

    def _update_weights(self,xi,target):
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_[1:] += self.lr * error * xi
        self.w_[0] += self.lr * error
        cost=error**2
        return cost

    def _init_weights(self,m):
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized=True

    def _shufle(self,X,y):
        r=self.rgen.permutation(len(y))
        return X[r],y[r]

    def activation(self,X):
        return X

    def net_input(self,X):
        return np.dot(X,self.w_[1:])+self.w_[0]

    def predict(self,X):
        return np.where(self.net_input(X)>=0.0, 1, -1)



def plot_desicion_regions(X,y,ppn,resolution=0.02):
    markers=('x','o','4')
    colors=('red','blue','green')
    cmap=ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1

    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

    Z=ppn.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z=Z.reshape(xx2.shape)

    plt.contourf(xx1,xx2,Z,cmap=cmap,alpha=0.4)

    for ix,cl in enumerate(np.unique(y)):
        plt.scatter(X[y==cl,0],
                    X[y==cl,1],
                    color=colors[ix],
                    marker=markers[ix],
                    edgecolors='black',
                    )
    plt.show()

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

df=pd.read_csv(url,header=None,encoding='utf-8')

X=df.iloc[0:100, [0,2] ].values
y=df.iloc[0:100, 4 ].values
y=np.where(y=='Iris-setosa',1,-1)

X_std=np.copy(X)
X_std[:,0] = (X[:,0]-X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1]-X[:,1].mean()) / X[:,1].std()

# plt.scatter(X[0:50,0],X[0:50,1],color='red',label='Iris-setosa',marker='x')
# plt.scatter(X[50:100,0],X[50:100,1],color='blue',label='Iris-versicolor',marker='o')
# plt.legend(loc='upper left')
# plt.xlabel('feature1')
# plt.ylabel('feature2')
# plt.show()
# plt.close()

n_iter=50
ppn=Perceptron(lr=0.01,n_iter=n_iter)
ppn=ppn.fit(X_std,y)

# print(ppn.cost_)
plt.plot([n for n in range(n_iter)],ppn.cost_,marker='o')
plt.show()

plot_desicion_regions(X_std,y,ppn)




