import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
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


for depth in range(1,12):
    dt=DecisionTreeClassifier(criterion='gini',max_depth=depth,random_state=1)
    dt.fit(X_train,y_train)
    # plot_tree(dt)
    # plt.show()
    # plt.close()
    plot_desicion_regions(X_test,y_test,dt)
    plt.title(f'Decision Tree depth={depth}')
    plt.show()
    plt.close()