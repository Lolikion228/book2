import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
def plot_desicion_regions(X,y,ppn,resolution=0.02,test_idx=None):
    markers=('x','o','s','v','^')
    colors=('red','blue','green','gray','cyan')
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
                    label=cl
                    )

    plt.legend(loc='upper left')
    plt.show()
    plt.close()

np.random.seed(1)
X_xor=np.random.randn(200,2)
y_xor=np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)
y_xor=np.where(y_xor,1,-1)


# plt.scatter(X_xor[np.where(y_xor==1),0],X_xor[np.where(y_xor==1),1],marker='x',color='blue')
# plt.scatter(X_xor[np.where(y_xor==-1),0],X_xor[np.where(y_xor==-1),1],marker='o',color='red')
# plt.show()
# plt.close()

svm=SVC(kernel='rbf',random_state=1,C=1,gamma=100)
svm.fit(X_xor,y_xor)
plot_desicion_regions(X_xor,y_xor,svm)

LogReg=LogisticRegression(solver='lbfgs',random_state=1,multi_class='ovr')
LogReg.fit(X_xor,y_xor)
plot_desicion_regions(X_xor,y_xor,LogReg)