import pandas as pd
from io import StringIO
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
""" #1
csv_data= \
'''
A,B,C,D
,,,4.0
,6.0,,8.0
10.0,11.0,12.0,
20,25,30,-10
10,20,30,40
'''
imr=SimpleImputer(missing_values=np.nan,strategy='mean')
df=pd.read_csv(StringIO(csv_data))
imr.fit(df.values)
imputed_data=imr.transform(df.values)
print(df)
print(imputed_data)
print(df.isnull().sum())
df1=df.dropna(axis=0,thresh=1)
df2=df.dropna(axis=0,subset=['B','D'])
print()
print(df2)
print()
print(df2)
"""



""" #2
df=pd.DataFrame([
    ['green','M','10.1','class2'],
    ['red','L','13.5','class1'],
    ['blue','XL','15.3','class2']
])
# print(df)
df.columns=['color','size','price','class_label']
print(df)

df['price']=df['price'].astype(float)

map_size={'XL':3,'L':2,'M':1}
inv_map_size={map_size[key]:key for key in map_size.keys() }
df['size']=df['size'].map(map_size)
print(df)

class_mapping={cl:ix for ix,cl in enumerate(np.unique(df['class_label']))}
inv_class_mapping={v:k for k,v in class_mapping.items()}
df['class_label']=df['class_label'].map(class_mapping)
print(df)

X=df[['color','size','price']].values
ohe=OneHotEncoder(drop='first',categories='auto')
print(X)
# X=ohe.fit_transform(X[:,0].values.reshape(-1,1)).toarray()
ct=ColumnTransformer([
    ('onehot',ohe,[0]),
    ('Nada','passthrough',[1,2])
])
print(ct.fit_transform(X).astype(float))

print(pd.get_dummies(df[['color','size','price']],drop_first=True,dtype='int'))
"""



df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/wine/wine.data',
header=None)

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
# print(df_wine.head(5))

X,y=df_wine.values[ : ,1:],df_wine.values[  : ,0]


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y)

mms=MinMaxScaler()
std=StandardScaler()
# print(X_train[ 0:5 : ,1:3 ])
X_train_norm=mms.fit_transform(X_train)
X_test_norm=mms.transform(X_train)
X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)
# print(X_train_norm[ 0:5 : ,1:3 ])
# print(X_train_std[ 0:5 : ,1:3 ])


""""  Regularization power
fig=plt.figure()
ax=plt.subplot(111)
colors = [ 'blue', 'green', 'red', 'cyan',
'magenta', 'yellow', 'black',
'pink', 'lightgreen', 'lightblue',
'gray', 'indigo', 'orange']

weights,params=[],[]
for c in range(-4,6):
    lr = LogisticRegression(penalty="l1", C=10**c, solver="liblinear", random_state=0, multi_class='ovr',max_iter=100)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

weights=np.array(weights)
for column,color in zip(range(weights.shape[1]),colors):
    plt.plot(params,weights[:,column],color=color,label=df_wine.columns[column+1])

plt.xlim( [10** (-5), 10**5] )
plt.legend()
plt.xscale('log')
ax.legend(loc='lower left')
plt.show()
"""




""" Sequential Backward Selection
class SBS():
    def __init__(self,estimator,k_features,scoring=accuracy_score,test_size=0.25,random_state=0):
        self.estimator = estimator
        self.k_features = k_features
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state
        self.best_k_features=999999
        self.best_score=0
    def fit(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(
            X,y,test_size=self.test_size,
            stratify=y,random_state=self.random_state
        )

        dim=X.shape[1]
        self._ixs=tuple(range(dim))
        self.scores_=[]
        self.subsets_=[self._ixs]
        self.scores_.append(self._calc_score(X_train,y_train,X_test,y_test,self._ixs))
        if self.scores_[-1]>=self.best_score:
            self.best_score=self.scores_[-1]
            self.best_k_features=dim
        while dim>self.k_features:
            scores = []
            subsets = []
            for p in combinations(self._ixs,r=dim-1):
                scores.append(self._calc_score(X_train,y_train,X_test,y_test,p))
                subsets.append(p)

            best_ix=np.argmax(scores)
            self._ixs=subsets[best_ix]
            self.subsets_.append(self._ixs)
            self.scores_.append(scores[best_ix])
            if self.scores_[-1] >= self.best_score:
                self.best_score = self.scores_[-1]
                self.best_k_features = dim-1
            dim -= 1
    def transform(self,X):
        return X[:,list(self.subsets_[-self.best_k_features])]
    def _calc_score(self,X_train,y_train,X_test,y_test,ixs):
        self.estimator.fit(X_train[:,ixs],y_train)
        y_pred=self.estimator.predict(X_test[:,ixs])
        return self.scoring(y_test,y_pred)

X_train_std_1=np.copy(X_train_std)
X_test_std_1=np.copy(X_test_std)
knn=KNeighborsClassifier(n_neighbors=5)
sbs=SBS(knn,4,random_state=0)
sbs.fit(X_train_std_1,y_train)
X_train_std_1=sbs.transform(X_train_std_1)
X_test_std_1=sbs.transform(X_test_std_1)

print(sbs.scores_)
print(sbs.best_k_features,sbs.best_score)
plt.plot([k for k in range(len(sbs.scores_))],sbs.scores_,marker='o')
plt.xlabel('feature_cnt')
plt.ylabel('accuracy')
plt.show()
k3=list( sbs.subsets_[-sbs.best_k_features] )
print(df_wine.columns[1:][k3])

knn.fit(X_train_std,y_train)
print(knn.score(X_test_std,y_test))
knn.fit(X_train_std_1,y_train)
print(knn.score(X_test_std_1,y_test))
"""



""" RandomForest Feature Selection
tree=RandomForestClassifier(n_estimators=1000)
tree.fit(X_train,y_train)
important=tree.feature_importances_
important=[ [num,ix] for ix,num in enumerate(important)]
important.sort(reverse=True)
important=np.array(important)
for imp,ix in important:
    print(df_wine.columns[1:][int(ix)],'importance:',imp)

# t=np.array(list(map(int,important[ : ,1])))
# print(t)
# plt.bar(df_wine.columns[1:][t],important[:,0])
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()
"""