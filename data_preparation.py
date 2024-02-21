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


""" pRINCIPIAL COMPONENT aNALYSIS
# cov_mat=np.cov(X_train_std.T)
# eigen_vals,eigen_vecs=np.linalg.eig(cov_mat)
# tot=np.sum(eigen_vals)
# var_exp=[ (i/tot) for i in sorted(eigen_vals,reverse=True) ]
# cum_var_exp=np.cumsum(var_exp)
# plt.bar(range(1,14),cum_var_exp,label='куммулятивная объяснённая дисперсия')
# plt.bar(range(1,14),var_exp,label='индивидуальная объяснённая дисперсия',align='center')
# plt.ylabel('Коэффициент объясненной дисперсии')
# plt.xlabel ('Индекс главного компонента')
# plt.legend(loc='best')
# plt.tight_layout()
# plt. show ()

# eigen_pairs = [  (np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals)) ]
# eigen_pairs.sort(key=lambda k: k[0],reverse=True)
# feats=2
# w=np.hstack( tuple(eigen_pairs[i][1][:,np.newaxis] for i in range(feats))  )
# # print(w)
# # print(X_train_std.shape,w.shape)
# X_train_std_transformed=np.dot(X_train_std,w)
# # print(X_train_std_transformed.shape,y_train.shape)
# colors=['red','blue','green']
# markers=['s','x','o']
# for i,c,m in zip(np.unique(y_train),colors,markers):
#     plt.scatter(X_train_std_transformed[y_train==i, 0],
#                 X_train_std_transformed[y_train==i, 1],
#                 c=c,
#                 marker=m,
#                 label=i)
# 
# plt.legend('lower left')
# plt.tight_layout()
# plt.show()
"""

"""
### PCA from sklearn
from sklearn.decomposition import PCA
from plot_regions import plot_desicion_regions
pca=PCA(n_components=4)
lr=LogisticRegression(multi_class='ovr',solver='lbfgs')
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)
# lr.fit(X_train_pca,y_train)
# print(lr.score(X_test_pca,y_test))
# lr.fit(X_train_std,y_train)
# print(lr.score(X_test_std,y_test))
# plot_desicion_regions(X_train_pca,y_train,lr)
# plt.title('train_set_pca')
# plt.show()
# plt.close()
# plot_desicion_regions(X_test_pca,y_test,lr)
# plt.title('test_set_pca')
# plt.show()
# plt.close()
"""



"""
### LDA Linear Discriminant Analysis
mean_vecs=[]

for i in range(1,4):
    mean_vecs.append(np.mean(X_train_std[y_train==i],axis=0))

d=13#features cnt
S_W=np.zeros((d,d))

# for label,mv in zip(range(1,4),mean_vecs):
#     s_i=np.zeros((d,d))
#     for row in X_train_std[y_train==label]:
#         row,mv=row.reshape(d,1),mv.reshape(d,1)
#         s_i+=np.dot( row-mv, (row-mv).T )
#     S_W+=s_i

for label in range(1,4):
    S_W+=np.cov(X_train_std[y_train==label].T)



S_B=np.zeros((d,d))
mean_all=np.mean(X_train_std,axis=0)
mean_all=mean_all.reshape(d,1)
for c,mv in zip(range(1,4),mean_vecs):
    mv=mv.reshape(d,1)
    S_B+=np.count_nonzero(y_train==c)*np.dot(mv-mean_all, (mv-mean_all).T)


eigenvalues,eigenvectors=np.linalg.eig( (np.linalg.inv(S_W).dot(S_B)) )
# eigenvalues=eigenvalues.astype(float)
# eigenvectors=eigenvectors.astype(float)


eigenpairs = [ (np.abs(eigenvalues[i]),eigenvectors[:,i]) for i in range(len(eigenvalues))]
eigenpairs.sort(key=lambda k: k[0],reverse=True)


w=np.hstack( tuple(eigenpairs[i][1][:,np.newaxis].real for i in range(2))  )

# print(w)

X_train_std_LDA=np.dot(X_train_std,w)
colors=['red','green','blue']
markers=['s','o','x']

for ix,c,m in zip(range(1,4),colors,markers):
    plt.scatter(X_train_std_LDA[y_train==ix,0],
                X_train_std_LDA[y_train==ix,1],
                c=c,
                marker=m,
                label=ix)
plt.title('LDA')
plt.show()
plt.close()

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
"""





"""
### Kernelized PCA
from scipy.spatial.distance import pdist,squareform
from sklearn.datasets import make_moons,make_circles
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler as SS
from scipy.linalg import eigh
from sklearn.decomposition import KernelPCA

# def rbf_kernel_pca(X,gamma,n_components):
#     sq_dists=pdist(X,metric='sqeuclidean')
#     mat_sq_dists=squareform(sq_dists)
#     K=np.exp(-gamma*mat_sq_dists)
#     N=K.shape[0]
#     one_n=np.ones((N,N))/N
#     K=K-one_n.dot(K)-K.dot(one_n)+ (one_n.dot(K)).dot(one_n)
#     eigvals,eigvecs=eigh(K)
#     eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
#     alphas=np.column_stack(tuple(eigvecs[:,i] for i in range(n_components)))
#     lambdas=[eigvals[i] for i in range(n_components)]
#     return alphas,lambdas

def rbf_kernel_pca(X, gamma, n_components):

    sq_dists = pdist (X, 'sqeuclidean' )
    mat_sq_dists = squareform(sq_dists)
    K= np.exp (-gamma * mat_sq_dists)

    N = K.shape[0]
    one_n = np.ones( (N,N)) / N
    K= K - one_n.dot (K) - K.dot (one_n) + one_n.dot (K) .dot (one_n)
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
    alphas = np.column_stack([eigvecs[:, i]
    for i in range(n_components)])
    lambdas= [eigvals [i] for i in range (n_components)]
    return alphas,lambdas

# def project_x(x_new,X,gamma,alphas,lambdas):
#     pair_dist=np.array([   np.sum((x_new-row)**2) for row in X ])
#     k=np.exp(-gamma*pair_dist)
#     # print(pair_dist.shape)
#     return k.dot(alphas/lambdas)
def project_x (x_new, X, gamma, alphas, lambdas):
    pair_dist = np.array( [np.sum(
    (x_new-row) **2) for row in X])
    k = np.exp (-gamma * pair_dist)
    # print(k.shape,alphas.shape,len(lambdas))
    # print(alphas[0:3])
    # print(lambdas)
    # print((alphas/lambdas)[:3])
    return k.dot (alphas / lambdas)

# X,y=make_moons(n_samples=100)
X,y=make_circles(n_samples=100,random_state=123,noise=0.1,factor=0.3)
# ss=SS()
# X_std=ss.fit_transform(X)


# plt.scatter(X[y==0,0],X[y==0,1],marker='o',c='red')
# plt.scatter(X[y==1,0],X[y==1,1],marker='x',c='blue')
# plt.title('Raw')
# plt.show()
# plt.close()

# plt.scatter(X_std[y==0,0],X_std[y==0,1],marker='o',c='red')
# plt.scatter(X_std[y==1,0],X_std[y==1,1],marker='x',c='blue')
# plt.title('Raw_std')
# plt.show()
# plt.close()



# pca=PCA(n_components=2)
# X1=pca.fit_transform(X,y)
# X1_std=pca.fit_transform(X_std,y)
# plt.scatter(X1[y==0,0],X1[y==0,1],marker='o',c='red')
# plt.scatter(X1[y==1,0],X1[y==1,1],marker='x',c='blue')
# plt.title('PCA')
# plt.show()
# plt.close()
#
# plt.scatter(X1_std[y==0,0],X1_std[y==0,1],marker='o',c='red')
# plt.scatter(X1_std[y==1,0],X1_std[y==1,1],marker='x',c='blue')
# plt.title('PCA_std')
# plt.show()
# plt.close()

#{'rbf', 'precomputed', 'sigmoid', 'poly', 'cosine', 'linear'}

gamma=5
x_new=X[45]
print('x_new=',x_new)
kpca=KernelPCA(n_components=2,kernel='rbf',gamma=gamma)
X22=kpca.fit_transform(X)
alphas1,lambdas1=rbf_kernel_pca(X,gamma,2)
x_proj=alphas1[45]
x_proj2=X22[45]
print('x_proj=',x_proj)
print('x_proj2=',x_proj2)
x_reproj=project_x(x_new,X,gamma,alphas1,lambdas1)
print('x_reproj=',x_reproj)
print('x_reproj2=',kpca.transform(x_new.reshape(1,2)))


# X2_std=rbf_kernel_pca(X_std,25,2)
# plt.scatter(X2[y==0,0],X2[y==0,1],marker='o',c='red')
# plt.scatter(X2[y==1,0],X2[y==1,1],marker='x',c='blue')
# plt.title(f'my_KPCA gamma={gamma}')
# plt.show()
# plt.close()


# plt.scatter(X22[y==0,0],X22[y==0,1],marker='o',c='red')
# plt.scatter(X22[y==1,0],X22[y==1,1],marker='x',c='blue')
# plt.title(f'sklearn_KPCA gamma={gamma}')
# plt.show()
# plt.close()
# plt.scatter(X2_std[y==0,0],X2_std[y==0,1],marker='o',c='red')
# plt.scatter(X2_std[y==1,0],X2_std[y==1,1],marker='x',c='blue')
# plt.title('KPCA_std')
# plt.show()
# plt.close()
"""



