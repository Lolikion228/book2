from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



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


iris=datasets.load_iris()
X=iris['data'][:,[2,3]]
y=iris['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)

sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)


svm=SVC(C=20.0,kernel='linear',random_state=1)
svm.fit(X_train_std,y_train)
plot_desicion_regions(X_test_std, y_test, svm)

# ppn=Perceptron(eta0=0.1,random_state=1)
# ppn.fit(X_train_std,y_train)
# weights=[]
# for c in range(25,30):
# LogReg=LogisticRegression(solver='lbfgs',random_state=1,multi_class='ovr')
# LogReg.fit(X_train_std,y_train)
# plot_desicion_regions(X_test_std, y_test, LogReg)
    # weights.append(list(LogReg.coef_[1]))
# print(weights)
# weights=np.array(weights)
#
# plt.plot(np.arange(-10,10), weights[:,0])
# plt.plot(np.arange(-10,10),weights[:,1],linestyle='--')
# plt.show()
# y_pred1=ppn.predict(X_test_std)
# y_pred2=LogReg.predict(X_test_std)



# plot_desicion_regions(X_test_std,y_test,ppn)
# plot_desicion_regions(X_test_std,y_test,LogReg)
# preds=LogReg.predict_proba(X_test_std[:10])
# for i,x in enumerate(preds):
#     print(x,f'pred={np.argmax(x)}',f'true={y_test[i]}','sum=',x.sum())

# print(ppn.score(X_test_std,y_test))
# print(np.where(y_pred==y_test,1,0).sum(),len(y_test))