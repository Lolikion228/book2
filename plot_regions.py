
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def plot_desicion_regions(X,y,classifier,resolution=0.02):
    markers=('x','o','s','v','^')
    colors=('red','blue','green','gray','cyan')
    cmap=ListedColormap(colors[:len(np.unique(y))])
    x1_min,x1_max=X[:,0].min()-1,X[:,0].max()+1
    x2_min,x2_max=X[:,1].min()-1,X[:,1].max()+1

    xx1,xx2=np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))

    Z=classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
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