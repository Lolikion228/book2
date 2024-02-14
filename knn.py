import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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



iris=datasets.load_iris()
X=iris['data'][:,[2,3]]
y=iris['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)


for p in range(1,100,10):
    rf=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=p)
    rf.fit(X_train_std,y_train)
    plot_desicion_regions(X_train_std,y_train,rf)
    plt.title(f'Train_set n_neighbors={p}')
    plt.show()
    plt.close()
    plot_desicion_regions(X_test_std,y_test,rf)
    plt.title(f'Test_set n_neighbors={p}')
    plt.show()
    plt.close()

