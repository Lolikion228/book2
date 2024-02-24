import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split,cross_val_score
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier
from sklearn.decomposition import PCA,KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


df=pd.read_csv('spaceship-titanic/train.csv')

y=df['Transported'].map(int).values

df=df.drop(columns=['Transported'])
# print(df.columns)


#feature 1
df=df.drop(columns=["PassengerId"])


#Feature 2   OH
si2=SimpleImputer(missing_values=3,strategy='most_frequent')
le2=LabelEncoder()
df['HomePlanet']=le2.fit_transform(df['HomePlanet'])
df['HomePlanet']=si2.fit_transform(df['HomePlanet'].values.reshape(-1,1))
# print(set(le2.inverse_transform(df['HomePlanet'])))


#Feature 3
si3=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df['CryoSleep']=df['CryoSleep'].map(int,na_action='ignore')
df['CryoSleep']=si3.fit_transform(df['CryoSleep'].values.reshape(-1,1))
df['CryoSleep']=df['CryoSleep'].astype(int)
# print(set(df['CryoSleep'].values))



#Feature 5 OH
si5=SimpleImputer(missing_values=3,strategy='most_frequent')
le5=LabelEncoder()
df['Destination']=le5.fit_transform(df['Destination'])
df['Destination']=si5.fit_transform(df['Destination'].values.reshape(-1,1))
# print(set(le5.inverse_transform(df['Destination'])))


#Feature 6
si6=SimpleImputer(missing_values=np.nan,strategy='mean')
df['Age']=si6.fit_transform(df['Age'].values.reshape(-1,1))


#Feature 7
si7=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df['VIP']=df['VIP'].map(int,na_action='ignore')
df['VIP']=si7.fit_transform(df['VIP'].values.reshape(-1,1))
df['VIP']=df['VIP'].astype(int)
# print(set(df['VIP'].values))


#Feature 8-12
si8=SimpleImputer(missing_values=np.nan,strategy='mean')
columns=(df.columns[i] for i in range(6,11))
for column in columns:
    df[column] = si8.fit_transform(df[column].values.reshape(-1, 1))
# for x in df.columns[6:11]:
#     print(df.isnull()[x].sum())

#Feature 13
le13_1=LabelEncoder()
le13_2=LabelEncoder()
data=[]
for i in range(len(df['Name'].values)):
    if type(df['Name'].values[i])==str:
        data.append(df['Name'].values[i].split())
    else:
        data.append([df['Name'].values[i],df['Name'].values[i]])
data=np.array(data)

fn=[]
ln=[]
for x in data:
    fn.append(x[0])
    ln.append(x[1])
fn=np.array(fn)
ln=np.array(ln)
df=df.drop(columns=['Name'])
df.insert(11,"FName",fn,True)
df.insert(12,"LName",ln,True)


df["FName"]=le13_1.fit_transform(df["FName"])
si13_1=SimpleImputer(missing_values=le13_1.transform(['nan'])[0],strategy='most_frequent')
df["FName"]=si13_1.fit_transform(df["FName"].values.reshape(-1,1))

df["LName"]=le13_2.fit_transform(df["LName"])
si13_2=SimpleImputer(missing_values=le13_2.transform(['nan'])[0],strategy='most_frequent')
df["LName"]=si13_2.fit_transform(df["LName"].values.reshape(-1,1))



#Feature 4  OH
le4_1=LabelEncoder()
le4_3=LabelEncoder()
data=[]
for i in range(len(df['Cabin'].values)):
    if type(df['Cabin'].values[i])==str:
        data.append(df['Cabin'].values[i].split('/'))
    else:
        data.append([df['Cabin'].values[i],df['Cabin'].values[i],df['Cabin'].values[i]])
data=np.array(data)

f1=[]
f2=[]
f3=[]
for x in data:
    f1.append(x[0])
    f2.append(x[1])
    f3.append(x[2])


f1=np.array(f1)
df.insert(13,"f1",f1,True)
df["f1"]=le4_1.fit_transform(df["f1"])
si4_1=SimpleImputer(missing_values=le4_1.transform(['nan'])[0],strategy='most_frequent')
df["f1"]=si4_1.fit_transform(df["f1"].values.reshape(-1,1))
# df['f1']=le4_1.inverse_transform(df['f1'])


f2=np.array(f2)
df.insert(14,"f2",f2,True)
def map1(x):
    if x=='nan': return -1
    else: return int(x)
df['f2']=df['f2'].map(map1)
si4_2=SimpleImputer(missing_values=-1,strategy='most_frequent')
df["f2"]=si4_2.fit_transform(df["f2"].values.reshape(-1,1))


f3=np.array(f3)
df.insert(15,"f3",f3,True)
df["f3"]=le4_3.fit_transform(df["f3"])
si4_3=SimpleImputer(missing_values=le4_3.transform(['nan'])[0],strategy='most_frequent')
df["f3"]=si4_3.fit_transform(df["f3"].values.reshape(-1,1))
# df['f3']=le4_3.inverse_transform(df['f3'])

df=df.drop(columns=['Cabin'])

X=df.values




#Model
pca=PCA(n_components=15)
# lda=LinearDiscriminantAnalysis(n_components=1)
kpca=KernelPCA(n_components=15,kernel='rbf')

X=pca.fit_transform(X)

ss=StandardScaler()

X_std=ss.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)


X_train_std=np.copy(X_train)
X_train_std=ss.fit_transform(X_train_std)
X_test_std=np.copy(X_test)
X_test_std=ss.fit_transform(X_test_std)


lr=LogisticRegression(C=0.1,max_iter=10000,penalty='l2')
ksvm=SVC(kernel='rbf',gamma=0.1,C=1,max_iter=10000)
tree=DecisionTreeClassifier(criterion='entropy',max_depth=10)
bag=BaggingClassifier(estimator=tree,n_estimators=15)
ada=AdaBoostClassifier(estimator=tree,n_estimators=30,algorithm='SAMME')

model=ada

# print(ada.get_params())
#Normal train/test split
# model.fit(X_train,y_train)
# print(model.score(X_test,y_test))
# model.fit(X_train_std,y_train)
# print(model.score(X_test_std,y_test))


# KFold
# kfold=StratifiedKFold(n_splits=10).split(X_std,y)
# scores=[]
# for step,(train,test) in enumerate(kfold):
#     model.fit(X_std[train],y[train])
#     score=model.score(X_std[test],y[test])
#     scores.append(score)
# scores=np.array(scores)
# print(f'{model.__str__()[:model.__str__().index("(")]} accuracy: {scores.mean()} +- {scores.std()}')


#GridSearch
ensemble_param_grid={'n_estimators':np.linspace(3,100,20).astype(int),'estimator__max_depth':np.linspace(1,50,20).astype(int)}

lr_param_grid={'C':np.linspace(1e-5,100,100)}
ksvm_param_grid={'C':np.linspace(1e-5,100,10),'gamma':np.linspace(1e-5,100,10)}

gs=GridSearchCV(ada,param_grid=ensemble_param_grid,cv=5,scoring='accuracy',refit=True,n_jobs=4)
gs.fit(X_std,y)
print(gs.best_params_)
print(gs.best_score_)


#one hot encoding на неупорядоченные именные признаки


