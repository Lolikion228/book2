import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier
from sklearn.decomposition import PCA,KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

df=pd.read_csv('spaceship-titanic/train.csv')
df2=pd.read_csv('spaceship-titanic/test.csv')

def prepare_data(df):
    train = 0
    if 'Transported' in df.columns: train=1
    if train:
        y=df['Transported'].map(int).values
        df=df.drop(columns=['Transported'])
    # print(df.columns)


    #feature 1
    # p=df['PassengerId'].values
    # p=[int(x.split('_')[1])-1 for x in p]
    # df['PassengerId']=p
    # r=pd.get_dummies(df['PassengerId'])
    # for i in range(len(r.columns)):
    #     df.insert(loc=len(df.columns),column=2*str(r.columns[i]),value=r[r.columns[i]].values.astype(int))
    df=df.drop(columns=['PassengerId'])



    #Feature 2   Drop first??

    si2=SimpleImputer(missing_values=3,strategy='most_frequent')
    le2=LabelEncoder()
    df['HomePlanet']=le2.fit_transform(df['HomePlanet'])
    df['HomePlanet']=si2.fit_transform(df['HomePlanet'].values.reshape(-1,1))
    df['HomePlanet']=le2.inverse_transform(df['HomePlanet'])
    z=pd.get_dummies(df['HomePlanet'],dtype=int)
    df=df.drop(columns=['HomePlanet'])
    df.insert(loc=0,column="Earth",value=z['Earth'].values)
    df.insert(loc=1,column="Europa",value=z['Europa'].values)
    df.insert(loc=2,column="Mars",value=z['Mars'].values)



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
    df['Destination']=le5.inverse_transform(df['Destination'])
    z=pd.get_dummies(df['Destination'],dtype=int)
    df=df.drop(columns=['Destination'])
    df.insert(loc=3,column="55 Cancri e",value=z['55 Cancri e'].values)
    df.insert(loc=4,column="PSO J318.5-22",value=z['PSO J318.5-22'].values)
    df.insert(loc=5,column="TRAPPIST-1e",value=z['TRAPPIST-1e'].values)


    #Feature 6
    si6=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    df['Age']=si6.fit_transform(df['Age'].values.reshape(-1,1))


    #Feature 7
    si7=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    df['VIP']=df['VIP'].map(int,na_action='ignore')
    df['VIP']=si7.fit_transform(df['VIP'].values.reshape(-1,1))
    df['VIP']=df['VIP'].astype(int)
    # print(set(df['VIP'].values))


    #Feature 8-12
    si8=SimpleImputer(missing_values=np.nan,strategy='mean')
    # print(df.columns)
    columns=('RoomService','FoodCourt','ShoppingMall','Spa','VRDeck')
    for column in columns:
        df[column] = si8.fit_transform(df[column].values.reshape(-1, 1))
    # for x in df.columns[6:11]:
    #     print(df.isnull()[x].sum())

    #Feature 13
    # df=df.drop(columns=['Name'])
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
    df.insert(len(df.columns),"FName",fn,True)
    df.insert(len(df.columns),"LName",ln,True)


    df["FName"]=le13_1.fit_transform(df["FName"])
    si13_1=SimpleImputer(missing_values=le13_1.transform(['nan'])[0],strategy='most_frequent')
    df["FName"]=si13_1.fit_transform(df["FName"].values.reshape(-1,1))

    df["LName"]=le13_2.fit_transform(df["LName"])
    si13_2=SimpleImputer(missing_values=le13_2.transform(['nan'])[0],strategy='most_frequent')
    df["LName"]=si13_2.fit_transform(df["LName"].values.reshape(-1,1))



    #Feature 4  OH
    le4_1=LabelEncoder()
    le4_3=LabelEncoder()
    # print(df)
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
    df.insert(len(df.columns),"f1",f1,True)
    df["f1"]=le4_1.fit_transform(df["f1"])
    si4_1=SimpleImputer(missing_values=le4_1.transform(['nan'])[0],strategy='most_frequent')
    df["f1"]=si4_1.fit_transform(df["f1"].values.reshape(-1,1))
    # df['f1']=le4_1.inverse_transform(df['f1'])


    f2=np.array(f2)
    df.insert(len(df.columns),"f2",f2,True)
    def map1(x):
        if x=='nan': return -1
        else: return int(x)
    df['f2']=df['f2'].map(map1)
    si4_2=SimpleImputer(missing_values=-1,strategy='most_frequent')
    df["f2"]=si4_2.fit_transform(df["f2"].values.reshape(-1,1))


    f3=np.array(f3)
    df.insert(len(df.columns),"f3",f3,True)
    df["f3"]=le4_3.fit_transform(df["f3"])
    si4_3=SimpleImputer(missing_values=le4_3.transform(['nan'])[0],strategy='most_frequent')
    df["f3"]=si4_3.fit_transform(df["f3"].values.reshape(-1,1))


    df=df.drop(columns=['Cabin'])

    z=pd.get_dummies(df['f1'],dtype=int)
    df=df.drop(columns=['f1'])
    for i in range(len(z.columns)):
        df.insert(loc=len(df.columns),column=z.columns[i],value=z[z.columns[i]])






    # df=df.drop(columns=['Earth','TRAPPIST-1e',0])
    # df=df.drop(columns=['LName'])
    # df=df.drop(columns=['FName'])



    X = df.values
    if train:
        return X,y
    else:
        return X




#Model


X,y=prepare_data(df)
X2=prepare_data(df2)


pca=PCA(n_components=23)

# lda=LinearDiscriminantAnalysis(n_components=1)
# kpca=KernelPCA(n_components=23,kernel='rbf',n_jobs=-1,gamma=0.025)



# print('n_components=',pca.n_components_)
ss=StandardScaler()

X_std=ss.fit_transform(X)
X=pca.fit_transform(X)
X_std=pca.fit_transform(X_std)
# print(pca.n_components_)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)


X_train_std=np.copy(X_train)
X_train_std=ss.fit_transform(X_train_std)
X_test_std=np.copy(X_test)
X_test_std=ss.fit_transform(X_test_std)


lr=LogisticRegression(C=25,max_iter=100000,penalty='l2')
# ksvm=SVC(kernel='rbf',gamma=0.00047,C=250.0)
ksvm=SVC(kernel='poly',gamma=1,C=1)
tree=DecisionTreeClassifier(criterion='entropy',max_depth=8)
bag=BaggingClassifier(estimator=tree,n_estimators=30)
bag2=BaggingClassifier(estimator=ksvm,n_estimators=30)
ada=AdaBoostClassifier(estimator=tree,n_estimators=30,algorithm='SAMME')
ada2=AdaBoostClassifier(estimator=ksvm,n_estimators=20,algorithm='SAMME',learning_rate=0.1)

model=ksvm
"""
TRY POLYNOMIAL KERNEL
g=0.025 c=3 0.7915
g=0.001 C=100 0.7942
g=0.00054 c=250 0.7951233 +- 0.008802
g=0.00047 c=250 0.7952380 +- 0.008739
"""


pipe2=make_pipeline(StandardScaler(),
                    PCA(n_components=23),
                    # PCA(),
                    SVC(kernel='rbf')
                    )


# print(pipe2.get_params())
# print(ada.get_params())
# Normal train/test split
# model.fit(X_train_std,y_train)
# print(model.score(X_test_std,y_test))
# model.fit(X_std,y)
# print(model.score(X_std,y))
# pred=model.predict(ss.fit_transform(pca.transform(X2)))
# pred=model.predict(ss.transform(X2))#or fit transform????
# df2=pd.read_csv('spaceship-titanic/sample_submission.csv')
# df2['Transported']=pred.astype(bool)
# df2.to_csv('pred.csv',index=False)

# KFold
#just cross_val score
# model=pipe2
scores=cross_val_score(model,X_std,y,cv=5,scoring='accuracy',verbose=3,n_jobs=-1)
print(f'{model.__str__()[:model.__str__().index("(")]} accuracy: {scores.mean()} +- {scores.std()}')


#GridSearch
ensemble_param_grid={'n_estimators':np.linspace(1,300,300).astype(int),'estimator__max_depth':np.linspace(1,50,50).astype(int)}
lr_param_grid={'C':np.linspace(1e-5,100,2000)}
ksvm_param_grid={'C':np.linspace(1e-5,100,10),'gamma':np.linspace(1e-5,100,10)}

pipe1_param_grid={'pca__n_components':[i for i in range(1,X.shape[1]+1)],}

pipe2_param_grid={'pca__n_components':[i for i in range(1,X.shape[1]+1)]}

pipe2k_param_grid={'svc__C':np.linspace(248,252,5),
                    # 'pca__n_components':[25],
                   'svc__gamma':np.linspace(0.00007,0.00087,5)}

# gs=GridSearchCV(pipe2,param_grid=pipe2k_param_grid,cv=5,scoring='accuracy',n_jobs=-1,verbose=3)
# # rs=RandomizedSearchCV(pipe2,param_distributions=pipe2k_param_grid,cv=5,scoring='roc_auc',n_jobs=-1,verbose=3,n_iter=30)
# gs.fit(X,y)
# print(gs.best_params_)
# print(gs.best_score_)
# print(gs.cv_results_)








































