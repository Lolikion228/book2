import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import \
    train_test_split,StratifiedKFold,cross_val_score,\
    learning_curve,validation_curve
import matplotlib.pyplot as plt
df=pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)

random_state=1
le=LabelEncoder()
X=df.values[:, 2:]
y=df.values[:, 1]
y=le.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=random_state)
pipe_lr=make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(random_state=random_state,
                       solver="lbfgs",max_iter=10000)
)

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
