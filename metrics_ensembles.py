import numpy as np
import pandas as pd
import scipy.interpolate
import sklearn.feature_extraction
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import \
    train_test_split,StratifiedKFold,cross_val_score,\
    learning_curve,validation_curve,GridSearchCV
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,make_scorer,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


# df=pd.read_csv('https://archive.ics.uci.edu/ml/'
# 'machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
# random_state=1
# le=LabelEncoder()
# X=df.values[:, 2:]
# y=df.values[:, 1]
# y=le.fit_transform(y)
#
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=random_state)
# pipe_lr=make_pipeline(
#     StandardScaler(),
#     PCA(n_components=2),
#     LogisticRegression(random_state=random_state,
#                        solver="lbfgs",max_iter=10000)
# )
# pipe_svc=make_pipeline(StandardScaler(),SVC(random_state=1))

# pipe_lr.fit(X_train,y_train)
# y_pred=pipe_lr.predict(X_test)
# print(pipe_lr.score(X_test,y_test))




# kfold=StratifiedKFold(n_splits=10).split(X_train,y_train)
# scores=[]
# for k,(train,test) in enumerate(kfold):
#     pipe_lr.fit(X_train[train],y_train[train])
#     score=pipe_lr.score(X_train[test],y_train[test])
#     scores.append(score)
#     print(f'block:{k};   classes:{np.bincount(y_train)};  score:{score}')
# print(f'kfold accuracy: {np.mean(scores)} +- {np.std(scores)}')

# scores=cross_val_score(pipe_lr,X_train,y_train,cv=10,n_jobs=-1)
# print(f'kfold accuracy: {np.mean(scores)} +- {np.std(scores)}')


"""  
###learning_curves
train_sizes,train_scores,test_scores=\
    learning_curve(estimator=pipe_lr,X=X_train,y=y_train,
                   train_sizes=np.linspace(0.1,1.0,10),
                   cv=10,
                    n_jobs=1)

train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)

test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)

plt.plot(train_sizes,train_mean,color='blue',
         marker='o',markersize=5,label='train accuracy')

plt.fill_between(train_sizes,
                 train_mean+train_std,
                 train_mean-train_std,
                 color='blue',
                 alpha=0.15
                 )

plt.plot(train_sizes,test_mean,color='green',
         marker='s',markersize=5,label='validation accuracy',linestyle='--')

plt.fill_between(train_sizes,
                 test_mean+test_std,
                 test_mean-test_std,
                 color='green',
                 alpha=0.15
                 )
plt.grid()
plt.xlabel('train data cnt')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.03])
plt.show()
"""



"""  
###validation_curves
param_range=[10**exp for exp in range(-8,8)]
train_scores,test_scores=validation_curve(
    pipe_lr,X_train,y_train,
    param_range=param_range,
    param_name='logisticregression__C',
    cv=10
)

train_mean=np.mean(train_scores,axis=1)
train_std=np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)


plt.plot(param_range,train_mean,color='blue',
         marker='o',markersize=5,label='train accuracy')

plt.fill_between(param_range,
                 train_mean+train_std,
                 train_mean-train_std,
                 color='blue',
                 alpha=0.15
                 )

plt.plot(param_range,test_mean,color='green',
         marker='s',markersize=5,label='validation accuracy',linestyle='--')

plt.fill_between(param_range,
                 test_mean+test_std,
                 test_mean-test_std,
                 color='green',
                 alpha=0.15
                 )
plt.grid()
plt.xlabel('C regularization parameter')
plt.ylabel('accuracy')
plt.xscale('log')
plt.legend(loc='lower right')
plt.ylim([0.1,1.0])
plt.show()
"""


"""
#вложенная перекрёстная проверка
#and grid search

param_range=[10**exp for exp in range(-4,5)]
param_grid=[ {'svc__C':param_range,'svc__kernel':['linear']},
             {'svc__C':param_range,'svc__kernel':['rbf'], 'svc__gamma':param_range}]

gs=GridSearchCV(estimator=pipe_svc,
                scoring='accuracy',n_jobs=-1,
                cv=2,refit=True,
                param_grid=param_grid)

gs.fit(X_train,y_train)
print(gs.best_params_)
print(gs.best_score_)
print(gs.best_estimator_.score(X_test,y_test))

#вложенная перекрёстная проверка
print(np.mean(cross_val_score(estimator=gs,X=X_test,y=y_test,cv=5)))
"""


"""
#confmat
# pipe_svc.fit(X_train,y_train)
# y_pred=pipe_svc.predict(X_test)
# confmat=confusion_matrix(y_test,y_pred)
# print(confmat)
# fig,ax=plt.subplots(figsize = (3,3))
# ax.matshow(confmat,cmap=plt.cm.get_cmap('Blues'),alpha=0.3)
# for i in range(confmat.shape[0]):
#     for j in range(confmat.shape[1]):
#         ax.text(x=j,y=i,
#                 s=confmat[i,j],
#                 va='center',ha='center')
# plt.xlabel('predict')
# plt.ylabel('true')
# plt.show()

# param_range=[10**exp for exp in range(-2,3)]
# param_grid=[ {'svc__C':param_range,'svc__kernel':['linear']},
#              {'svc__C':param_range,'svc__kernel':['rbf'], 'svc__gamma':param_range}]
# scorer=make_scorer(f1_score,pos_label=0)
# gs=GridSearchCV(estimator=pipe_svc,scoring=scorer,n_jobs=-1,cv=10,param_grid=param_grid)
# gs.fit(X_train,y_train)
# print(gs.best_score_)
# print(gs.best_params_)
"""


"""
#receiver operating characteristic
from sklearn.metrics import roc_curve,auc
from scipy import interpolate

pipline_lr=make_pipeline( StandardScaler(),
                          PCA(n_components=2),
                          LogisticRegression(solver='lbfgs',
                                             C=100,
                                             random_state=1,
                                            )
                          )
X_train2=X_train[:,[4,14]]

cv=list(StratifiedKFold(n_splits=3).split(X_train,y_train))

fig=plt.figure(figsize=(7,5))
mean_tpr=0.0
mean_fpr=np.linspace(0,1,100)
all_tpr=[]

for i,(train,test) in enumerate(cv):
    pipline_lr.fit(X_train2[train],y_train[train])
    probas=pipline_lr.predict_proba(X_train2[test])

    fpr,tpr,thresholds= roc_curve(y_train[test],probas[:,1],pos_label=1)
    print(fpr,tpr)

    mean_tpr+=np.interp(mean_fpr,fpr,tpr)

    mean_tpr[0]=0.0
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr,label='ROC block %d (area= %0.2f)'%(i+1,roc_auc) )


plt.plot([0,1],[0,1],label='случайное угадывание',linestyle='--',color=(0.6,0.6,0.6))
plt.plot([0,0,1],[0,1,1],label='идеально',linestyle=':',color='black')


mean_tpr/=len(cv)
mean_tpr[-1]=1
mean_auc=auc(mean_fpr,mean_tpr)
plt.plot(mean_fpr,mean_tpr,'k--',label='средняя ROC (площадь= %0.2f)' % mean_auc, lw=2)


plt.legend(loc='lower right')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
"""


"""
#imbalanced data
X_imb=np.vstack((X[y==0],X[y==1][:40]))
y_imb=np.hstack((y[y==0],y[y==1][:40]))

print(np.mean(np.zeros(y_imb.shape[0])==y_imb)*100)
print(f'samples from class 1 initial cnt={y_imb[y_imb==1].shape[0]}')
X_upsampled,y_upsampled=resample(X_imb[y_imb==1],y_imb[y_imb==1],n_samples=X_imb[y_imb==0].shape[0],
                                 replace=True)
print(f'samples from class 1 cnt after resampling={y_upsampled[y_upsampled==1].shape[0]}')

X_bal=np.vstack((X[y==0],X_upsampled))
y_bal=np.hstack((y[y==0],y_upsampled))
print(np.mean(np.zeros(y_bal.shape[0])==y_bal)*100)
"""



""""
#ensembles #1
from scipy.special import comb
import math

def ensemble_error(n_classifier,error):
    k_start=math.ceil(n_classifier/2)
    probs=[comb(n_classifier,k) *
           (error**k) *
           ( (1-error)**(n_classifier-k))
           for k in range(k_start,n_classifier+1)]
    return sum(probs)


n_classifiers=100
plt.plot(np.linspace(0.0,1.0,100),
         [ensemble_error(n_classifiers,error) for error in np.linspace(0.0,1.0,100) ],
         label='ensemble error',linestyle='--'
         )

plt.plot(np.linspace(0.0,1.0,100),
         np.linspace(0.0,1.0,100),
         label='basic error',linestyle=':')
plt.title(f'Ensemble with {n_classifiers} classifiers error')
# plt.show()


ex=np.array([
    [0.9,0.1],
    [0.8,0.2],
    [0.4,0.6]
])

mean=np.average(ex,axis=0,weights=[0.2,0.2,0.6])
print(mean,np.argmax(mean))
"""

from sklearn.base import BaseEstimator,ClassifierMixin,clone
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators
import operator

class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,classifiers,vote='classlabel',weights=None):
        self.classifiers=classifiers
        self.named_classeifiers={ k:v for k,v in _name_estimators(classifiers)}
        self.vote=vote
        self.weights=weights

    def fit(self,X,y):
        self.labelenc_=LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_=self.labelenc_.classes_
        self.classifiers_=[]
        for clf in self.classifiers:
            fitted_clf=clone(clf).fit(X,self.labelenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self,X):
        if self.vote=='probability':
            maj_vote=np.argmax( self.predict_proba(X),axis=1)
        else:
            preds=np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote=np.apply_along_axis(
                lambda x:
                np.argmax(np.bincount(x,weights=self.weights)),
                axis=1,
                arr=preds
            )
        maj_vote=self.labelenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self,X):
        probas=np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba=np.average(probas,axis=0,weights=self.weights)
        return avg_proba

    def get_params(self,deep=True):
        if not deep:
            # return super(MajorityVoteClassifier).get_params()
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out=self.named_classeifiers.copy()
            for name,clf in self.named_classeifiers.items():
                for key,value in clf.get_params(deep=True).items():
                    out['%s__%s'%(name,key)]=value
            return out

from itertools import product
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve,auc
# iris=datasets.load_iris()
# X,y=iris.data[50:, [1,2]],iris.target[50:]
# le=LabelEncoder()
#
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,stratify=y,random_state=1)

# clf1=LogisticRegression(penalty='l2',C=0.001,solver='lbfgs',random_state=1)
# clf2=KNeighborsClassifier(n_neighbors=1,metric='minkowski',p=2)
# clf3=DecisionTreeClassifier(criterion='entropy',max_depth=1)
#
# pipe1=Pipeline([ ['sc',StandardScaler()],['clf',clf1]])
# pipe3=Pipeline([ ['sc',StandardScaler()],['clf',clf3]])
# clf_labels=['LogisticRegression','KNN','DecisionTree']
#
#
# my_clf=MajorityVoteClassifier(classifiers=[pipe1,clf2,pipe3])
# clf_labels+=['MajorityVoting']
# all_clf=[pipe1,clf2,pipe3,my_clf]

# for clf,label in zip(all_clf,clf_labels):
#     scores = cross_val_score(clf, X_train, y_train, cv=10, scoring='roc_auc')
#     print("ROC_AUC: %0.2f (+- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#
#
#
# colors=['blue','green','red','orange']
# linestyles=['--','-.','-',':']
# for clf,label,clr,ls in zip(all_clf,clf_labels,colors,linestyles):
#     y_pred=clf.fit(X_train,y_train).predict_proba(X_test)[:,1]
#     fpr,tpr,thresholds=roc_curve(y_test,y_pred,pos_label=2)
#     roc_auc=auc(fpr,tpr)
#     plt.plot(fpr,tpr,
#              color=clr,
#              linestyle=ls,
#              label='%s (auc=%0.2f)' % (label,roc_auc))
# plt.legend(loc='lower right')
# plt.xlabel('FPR')
# plt.ylabel('TPR')
# plt.plot([0,1],[0,1],linestyle='--',color='gray',linewidth=2)
# plt.grid(alpha=0.5)
# plt.tight_layout()
# plt.show()
# plt.close()


# sc=StandardScaler()
# X_train_std=sc.fit_transform(X_train)
# x_min=X_train_std[:,0].min()-1
# x_max=X_train_std[:,0].max()+1
# y_min=X_train_std[:,1].min()-1
# y_max=X_train_std[:,1].max()+1
#
# xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
# f,axarr=plt.subplots(nrows=2,ncols=2,sharex='col',sharey='row',figsize=(7,5))
#
#
# for idx,clf,tt in zip(product([0,1],[0,1]),all_clf,clf_labels):
#     clf.fit(X_train_std,y_train)
#     Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
#     Z=Z.reshape(xx.shape)
#     axarr[idx[0],idx[1]].contourf(xx,yy,Z,alpha=0.3)
#     axarr[idx[0],idx[1]].scatter(X_train_std[y_train==0,0],
#                                  X_train_std[y_train==0,1],
#                                  c='blue',
#                                  marker='^',
#                                  s=50)
#     axarr[idx[0],idx[1]].scatter(X_train_std[y_train==1,0],
#                                  X_train_std[y_train==1,1],
#                                  c='green',
#                                  marker='o',
#                                  s=50)
#     axarr[idx[0],idx[1]].set_title(tt)
# plt.show()


#
# prms=my_clf.get_params(deep=True)
# for x in prms:
#     print(x)

# params={'pipeline-2__clf__max_depth':[1,2],
#         'pipeline-1__clf__C':[0.001,0.1,100]}
# grid=GridSearchCV(my_clf,param_grid=params,scoring='roc_auc',cv=10)
# grid.fit(X_train,y_train)
#
# for r,_ in enumerate(grid.cv_results_['mean_test_score']):
#     print(" %0.3f +- %0.2f %r"%(
#         grid.cv_results_['mean_test_score'][r],
#         grid.cv_results_['std_test_score'][r]/2.0,
#         grid.cv_results_['params'][r],
#     ))
#
# print(grid.best_params_)
# print(grid.best_score_)


from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
df_wine=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)

df_wine.columns=[
'Class label', 'Alcohol',
'Malic acid ', 'Ash',
'Alcalinityofash', 'Magnesium',
'Total phenols ', 'Flavanoids',
'Nonflavanoid phenols',
'Proanthocyanins',
'Color intensity', 'Ние',
'OD280/0D315 of diluted wines',
'Proline']

df_wine=df_wine[df_wine['Class label']!=1]
y=df_wine['Class label'].values
X=df_wine[['Alcohol','OD280/0D315 of diluted wines']].values
le=LabelEncoder()
y=le.fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)


tree=DecisionTreeClassifier(criterion='entropy',max_depth=None,random_state=1)
bag=BaggingClassifier(estimator=tree,n_estimators=500,max_samples=1.0,max_features=1.0,
                      bootstrap=True,bootstrap_features=False)


tree.fit(X_train,y_train)
y_train_pred=tree.predict(X_train)
y_test_pred=tree.predict(X_test)

tree_train=accuracy_score(y_train,y_train_pred)
tree_test=accuracy_score(y_test,y_test_pred)

print('tree',tree_train,tree_test)


bag.fit(X_train,y_train)
y_train_pred=bag.predict(X_train)
y_test_pred=bag.predict(X_test)

tree_train=accuracy_score(y_train,y_train_pred)
tree_test=accuracy_score(y_test,y_test_pred)

print('bagging',tree_train,tree_test)


# sc=StandardScaler()
# X_test_std=sc.fit_transform(X_test)
# x_min=X_test_std[:,0].min()-1
# x_max=X_test_std[:,0].max()+1
# y_min=X_test_std[:,1].min()-1
# y_max=X_test_std[:,1].max()+1
#
# xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
# f,axarr=plt.subplots(nrows=1,ncols=2,sharex='col',sharey='row',figsize=(8,4))
#
# all_clf=[tree,bag]
# clf_labels=['tree','bag']
# for idx,clf,tt in zip([0,1],all_clf,clf_labels):
#     clf.fit(X_test_std,y_test)
#     Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
#     Z=Z.reshape(xx.shape)
#     axarr[idx].contourf(xx,yy,Z,alpha=0.3)
#     axarr[idx].scatter(X_test_std[y_test==0,0],
#                                  X_test_std[y_test==0,1],
#                                  c='blue',
#                                  marker='^',
#                                  s=50)
#     axarr[idx].scatter(X_test_std[y_test==1,0],
#                                  X_test_std[y_test==1,1],
#                                  c='green',
#                                  marker='o',
#                                  s=50)
#     axarr[idx].set_title(tt)
# plt.show()



from sklearn.ensemble import AdaBoostClassifier

tree2=DecisionTreeClassifier(criterion='entropy',max_depth=1,random_state=1)
ada=AdaBoostClassifier(estimator=tree2,n_estimators=500,learning_rate=0.1,random_state=1)
# tree.fit(X_train,y_train)

# y_train_pred=tree.predict(X_train)
# y_test_pred=tree.predict(X_test)
# tree_train=accuracy_score(y_train,y_train_pred)
# tree_test=accuracy_score(y_test,y_test_pred)
# print(tree_train,tree_test)

ada.fit(X_train,y_train)
y_train_pred=ada.predict(X_train)
y_test_pred=ada.predict(X_test)
tree_train=accuracy_score(y_train,y_train_pred)
tree_test=accuracy_score(y_test,y_test_pred)
print('ada',tree_train,tree_test)



