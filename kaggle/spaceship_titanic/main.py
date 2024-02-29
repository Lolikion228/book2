import pandas as pd
import numpy as np
import sklearn.ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder,Normalizer,RobustScaler,PolynomialFeatures,PowerTransformer
from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier,RandomForestClassifier,GradientBoostingClassifier

import xgboost as xgb
from sklearn.decomposition import PCA,KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# from keras import Sequential
# from tensorflow.keras.layers import Dense,Dropout
# import tensorflow.keras.optimizers as opt
df=pd.read_csv('spaceship-titanic/train.csv')
df2=pd.read_csv('spaceship-titanic/test.csv')


def prepare_data(df,verbose=0):
    train = 0
    if 'Transported' in df.columns: train=1
    if train:
        y=df['Transported'].map(int).values


    nans0=np.array(df.isnull().astype(int).values.sum(axis=1))
    # df.insert(loc=len(df.columns),column='nan_cnt',value=nans0)
    # df.insert(loc=len(df.columns),column='nan_norm',value=nans0/sum(nans0))

    #feature 1  ok
    p0=df['PassengerId'].values
    p=[int(x.split('_')[0]) for x in p0]


    df['PassengerId']=p
    d=dict(df['PassengerId'].value_counts())
    p=[d[x] for x in df['PassengerId']]
    p=np.array(p)
    df['PassengerId']=p
    df=df.rename(columns={"PassengerId":"Group size"})

    #
    if verbose:
        for i in set(df['Group size']):
            plt.bar(i,y[df['Group size']==i].sum())
        plt.ylabel('Transported')
        plt.xlabel('Group size')
        plt.show()
        plt.close()

    #Feature 2  ok

    le2=LabelEncoder()
    df['HomePlanet']=le2.fit_transform(df['HomePlanet'])
    si2 = SimpleImputer(missing_values=le2.transform([np.nan])[0], strategy='most_frequent')
    df['HomePlanet']=si2.fit_transform(df['HomePlanet'].values.reshape(-1,1))
    df['HomePlanet']=le2.inverse_transform(df['HomePlanet'])
    z=pd.get_dummies(df['HomePlanet'],dtype=int)

    distances = {"Earth": 0, "Europa": 628.3, "Mars":225}

    # distances = {"Earth": 0, "Europa": 6.641136e-11, "Mars":2.3783e-11}
    for _name in distances.keys():
        df[f"numeric__Home"] = df['HomePlanet'].apply(lambda x: distances.get(x, 0))



    df=df.drop(columns=['HomePlanet'])
    df=pd.concat([df,z],axis=1)

    if verbose:
        for planet in ['Earth','Europa','Mars']:
            plt.bar(planet,y[df[planet]==1].sum())
        plt.ylabel('Transported')
        plt.xlabel('HomePlanet')
        plt.show()
        plt.close()


    #Feature 3 ok
    # si3=KNNImputer()
    si3=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    df['CryoSleep']=df['CryoSleep'].map(int,na_action='ignore')
    df['CryoSleep']=si3.fit_transform(df['CryoSleep'].values.reshape(-1,1))
    df['CryoSleep']=df['CryoSleep'].astype(int)

    if verbose:
        plt.bar('False', y[df['CryoSleep'] == 0].sum())
        plt.bar('True', y[df['CryoSleep'] == 1].sum())
        plt.ylabel('Transported')
        plt.xlabel('CryoSleep')
        plt.show()
        plt.close()

    #Feature 5 ok




    le5=LabelEncoder()
    df['Destination']=le5.fit_transform(df['Destination'])
    si5 = SimpleImputer(missing_values=le5.transform([np.nan])[0], strategy='most_frequent')
    df['Destination']=si5.fit_transform(df['Destination'].values.reshape(-1,1))
    df['Destination']=le5.inverse_transform(df['Destination'])
    z=pd.get_dummies(df['Destination'],dtype=int)
    df=pd.concat([df,z],axis=1)


    distances = {"TRAPPIST-1e": 39.6, "55 Cancri e": 40, "PSO J318.5-22": 80}
    for _name in distances.keys():
        df[f"numeric__Destination"] = df['Destination'].apply(lambda x: distances.get(x, 0))



    df = df.drop(columns=['Destination'])
    if verbose:
        for planet in ['55 Cancri e', 'PSO J318.5-22', 'TRAPPIST-1e']:
            plt.bar(planet, y[df[planet] == 1].sum())
        plt.ylabel('Transported')
        plt.xlabel('DestinationPlanet')
        plt.show()
        plt.close()


    #Feature 6 ok
    # si6=KNNImputer()
    si6=SimpleImputer(missing_values=np.nan,strategy='median')
    df['Age']=si6.fit_transform(df['Age'].values.reshape(-1,1))
    df.insert(loc=len(df.columns),column='Age group',value=np.zeros(df.values.shape[0]))
    ranges = [[0,13], [14,17], [18,34], [35,60], [61, 79]]
    for i in range(0, len(ranges)):
        df.loc[(ranges[i][0] <= df['Age']) & (df['Age'] <= ranges[i][1]), 'Age group'] = i + 1
    # df['Age']=df['Age group'].values
    # df=df.drop(columns=['Age group'])

    if verbose:
    #     # age_min=0.0; age_max=79.0
        for i in range(0,len(ranges)):
            plt.bar(f'{ranges[i][0]}:{ranges[i][1]}', y[df['Age'] == i+1].sum())
        plt.ylabel('Transported')
        plt.xlabel('Age group')
        plt.show()
        plt.close()
    # df = df.drop(columns=['Age'])
    #



    #Feature 7 ok
    df['VIP']=df['VIP'].map(int,na_action='ignore')
    si7 = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df['VIP']=si7.fit_transform(df['VIP'].values.reshape(-1,1))
    df['VIP']=df['VIP'].astype(int)

    if verbose:
        plt.bar('False', y[df['VIP'] == 0].sum())
        plt.bar('True', y[df['VIP'] == 1].sum())
        plt.ylabel('Transported')
        plt.xlabel('VIP')
        plt.show()
        plt.close()



    #Feature 8-12 ok????
    si8=KNNImputer()
    # si8=SimpleImputer(missing_values=np.nan,strategy='mean')
    columns=['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']



    dummy=(df[columns]==0.0).astype(int)
    for name in columns:
        df.insert(loc=len(df.columns),column='dummy_'+name,value=dummy[name] )

    df.insert(loc=len(df.columns), column='service_cnt', value=dummy.values.sum(axis=1))

    df['service_cnt']=df['service_cnt'].apply(lambda x: int(x in [5]) )

    # for num in set(df['service_cnt']):
    #     plt.bar(num,y[df['service_cnt']==num].sum())
    # plt.show()


    for column in columns:
        df[column] = si8.fit_transform(df[column].values.reshape(-1, 1))



    sub=df[columns].values
    sub_sum=np.sum(sub,axis=1).reshape(-1,1)

    df.insert(loc=len(df.columns),column='Spended money',value=sub_sum)


    #min=0.0 max=35987.0

    if verbose:
        ranges = np.linspace(0,35987,11)

        for i in range(1,len(ranges)):
            plt.bar(f'{int(ranges[i-1])}:{int(ranges[i])}', y[ (df['Spended money']<=ranges[i]) & (ranges[i-1]<=df['Spended money']) ].sum())

        plt.ylabel('Transported')
        plt.xlabel('Spended money')
        plt.yscale('log')
        plt.xticks(rotation=45,fontsize=8)
        plt.tight_layout()
        plt.show()
        plt.close()
    # df.drop(columns=columns)
    # df=df.drop(columns=['Spended money'])#




    #Feature 13 ok

    #family size




    si13=SimpleImputer(missing_values=np.nan,strategy='constant',fill_value='bebra bebra')
    df['Name']=np.array(si13.fit_transform(df['Name'].values.reshape(-1,1))).reshape(-1)
    df['Name']=np.array([x.split()[1] for x in df['Name']])




    si13 = SimpleImputer(missing_values='bebra', strategy='most_frequent')
    df['Name'] = np.array(si13.fit_transform(df['Name'].values.reshape(-1, 1))).reshape(-1)

    d=dict(df['Name'].value_counts())
    df['Name']=np.array([d[x] for x in df['Name']])
    si13 = SimpleImputer(missing_values=218, strategy='constant',fill_value=3)
    df['Name'] = np.array(si13.fit_transform(df['Name'].values.reshape(-1, 1))).reshape(-1)
    df=df.rename(columns={'Name':'Family size'})

    df['Family size']=df['Family size'].apply(lambda x: 6<=x<=14  )


    if verbose:
        for i in set(df['Family size']):
            plt.bar(i,y[df['Family size']==i].sum())
        plt.xlabel('Family size')
        plt.ylabel('Transported')
        plt.show()
        plt.close()



    #Feature 4  ok



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
    df.insert(len(df.columns),"f1",f1,True)
    df["f1"]=le4_1.fit_transform(df["f1"])
    si4_1=SimpleImputer(missing_values=le4_1.transform(['nan'])[0],strategy='most_frequent')
    df["f1"]=si4_1.fit_transform(df["f1"].values.reshape(-1,1))


    f2=np.array(f2)
    df.insert(len(df.columns),"f2",f2,True)
    def map1(x):
        if x=='nan': return -1
        else: return int(x)
    df['f2']=df['f2'].map(map1)
    si4_2=SimpleImputer(missing_values=-1,strategy='most_frequent')
    df["f2"]=si4_2.fit_transform(df["f2"].values.reshape(-1,1))
    df = df.rename(columns={"f2": 'cabin[2]'})

    f3=np.array(f3)
    df.insert(len(df.columns),"f3",f3,True)
    df["f3"]=le4_3.fit_transform(df["f3"])
    si4_3=SimpleImputer(missing_values=le4_3.transform(['nan'])[0],strategy='most_frequent')
    df["f3"]=si4_3.fit_transform(df["f3"].values.reshape(-1,1))
    df=df.rename(columns={"f3":'cabin[3]'})

    z=pd.get_dummies(df['f1'],dtype=int)
    df=df.drop(columns=['f1'])
    for i in range(len(z.columns)):
        df.insert(loc=len(df.columns),column='cabin[0]='+str(z.columns[i]),value=z[z.columns[i]])

    df = df.drop(columns=['Cabin'])




    # ranges=np.linspace(0,1894,5).astype(int)

    ranges=[   0  ,315, 631   ,1162,  1894]
    df.insert(loc=len(df.columns),column='cabin_num_range',value=np.zeros(df.values.shape[0]))
    for i in range(1, len(ranges)):
        df.loc[(ranges[i - 1] <= df['cabin[2]']) & (df['cabin[2]'] <= ranges[i]), 'cabin_num_range'] = i


    if verbose:
        for i in range(1,len(ranges)):
            plt.bar(f'{ranges[i-1]}:{ranges[i]}', y[df['cabin_num_range'] == i].sum())

        plt.ylabel('Transported')
        plt.xlabel('Cabin number range')
        plt.xticks(fontsize=8, rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close()
        df=df.drop(columns=['cabin[2]'])
    # df=df.drop(columns=['cabin_num_range'])

    if train:
        df = df.drop(columns=['Transported'])


    # print(df.columns)


    columns_sqrt = [ 'Spended money','RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
                     'numeric__Destination'
                    ]


    df[columns_sqrt] = df[columns_sqrt].apply(np.sqrt)
    df['Group size'] = df['Group size'].apply(np.log)



    df=df[['cabin_num_range','cabin[3]','Spended money','service_cnt','dummy_VRDeck','dummy_ShoppingMall',
           'dummy_FoodCourt','dummy_RoomService','numeric__Destination','TRAPPIST-1e',
           '55 Cancri e', 'PSO J318.5-22','Mars','Europa','Earth','numeric__Home','Age',
           'VRDeck','Spa','ShoppingMall','FoodCourt','RoomService','VIP','CryoSleep',
           'cabin[0]=0', 'cabin[0]=1', 'cabin[0]=2', 'cabin[0]=3', 'cabin[0]=4',
            'cabin[0]=5', 'cabin[0]=6', 'cabin[0]=7']]


    X = df.values

    wtf=['Family size','Group size']


    #count of nans???
    #age or age group????
    #drop pco and/or mars??
    #remove service_cnt????(bad at gbc) (good at ksvm)
    #in service_cnt change the list????? only 5??? or remove mapping???
    # remove some cabin[0]????
    #change ranges for age-cabin???
    #change range to initial vals???


    # passenger id log_group_size 55.8
    #cryo sleep 71.828
    #Age 54.25
     #vip normalno 50+-
    # RoomService sqrt 66.6
   # FoodCourt 62.1
    #ShoppingMall   62.98
   # Spa sqrt 66.6
    # VrDeck 65.8
    #family size 51.57
    #numeric__Home 58.357
    # from Earth 58.3
    # from europa 57.4
    # from mars 50.2  drop it?
    # to cancri e 54.19
    # to pso 49 drop it?
    # to trappist 54
    # numeric_destination 53.8
    # dummy_RoomService 66
    # dummy_FoodCourt 61
    # dummy_ShoppingMall 62
    # dummy_Spa 66.6
    # dummy_VRDeck 65.8
    #service_cnt 72.8
    #Spended money sqrt 73.4
    #cabin[3] 55
    #cabin[0]=.....  56.4
    #cabin_num_range 56.4




    if train:
        return X,y
    else:
        return X






#Model

#try to classificate by just one single feature to get optimal data_preapration




# Create Bins for spended moneys???

#FoodCourt & RoomService together has really good classification ability

#VRDeck, Spa, RoomService have a good differentiation between classes.
# We can create a new feature that tells the total expenditure in the above three features.





# I add dummy features to just detect if some options have been chosen+
# I put every single value into its own numeric feature признак= человек выбрал k сервисов+
# I summarised all of services into numeric__sum_services and numeric__count_services+
#I will add some "distance from (Earth)?" in (light-years)???? as an numeric mearure of every plannet in home and destination+
#make was there nans in this row column+



X,y=prepare_data(df,verbose=0) # 1, 2 or 3
# X2=prepare_data(df2)





ss=StandardScaler()

pca=PCA(n_components='mle')
# kpca=KernelPCA(kernel='rbf',n_components=30,gamma=0.004) #0.01 86 79.1

X_std=ss.fit_transform(X)

# print(pca.n_components_,X.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=0.2,stratify=y)



gbc=GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.08, subsample=0.6)
ksvm=SVC(kernel='rbf',gamma=0.0045,C=200) #0.0015 200 80.00764%
rf=RandomForestClassifier(n_estimators=200,criterion='gini',max_depth=5)


# lr=LogisticRegression(C=20,max_iter=100000,penalty='l2')
# tree=DecisionTreeClassifier(criterion='gini',max_depth=6)
# knn=KNeighborsClassifier(n_neighbors=3)
# bag=BaggingClassifier(estimator=tree,n_estimators=31)
# ada=AdaBoostClassifier(estimator=tree,n_estimators=30,algorithm='SAMME')
# xgb_params={'colsample_bytree': 0.8498791800104656, 'learning_rate': 0.020233442882782587, 'max_depth': 4, 'n_estimators': 469, 'subsample': 0.746529796772373}
# xgb_params={ 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200,'colsample_bytree': 0.8,'subsample': 0.7}
# xgb1=xgb.XGBClassifier(**xgb_params)
#
models=[gbc,ksvm]
# model=gbc

#
# model.fit(X_train,y_train)
# print(model.score(X_test,y_test))
# model.fit(X_std,y)
# print(model.score(X_std,y))
# pred=model.predict(ss.fit_transform(X2))#or fit transform????
# df2=pd.read_csv('spaceship-titanic/sample_submission.csv')
# df2['Transported']=pred.astype(bool)
# df2.to_csv('pred.csv',index=False)

# KFold
# model=pipe2
for model in models:
    scores=cross_val_score(model,X_std,y,cv=8,scoring='accuracy',n_jobs=-1)
    print(f'{model.__str__()[:model.__str__().index("(")]} accuracy: {round(scores.mean(),6)} +- {round(scores.std(),5)}')

# scores=cross_val_score(model,X_std,y,cv=8,scoring='accuracy',n_jobs=-1)
# print(f'{model.__str__()[:model.__str__().index("(")]} accuracy: {round(scores.mean(),6)} +- {round(scores.std(),5)}')


# print(np.argsort(model.feature_importances_))


# #GridSearch
# ensemble_param_grid={'n_estimators':np.linspace(50,250,100).astype(int),
#                      'max_depth':np.linspace(2,10,8).astype(int),
#                      'learning_rate':np.linspace(0.05,0.5,40)}

# lr_param_grid={'C':np.linspace(1e-5,100,2000)}
# ksvm_param_grid={'C':np.linspace(1e-5,100,10),'gamma':np.linspace(1e-5,100,10)}
# ksvm_param_grid={'C':[5,10,15,20],'gamma':np.linspace(0.001,0.01,30)}

# #
# pipe2k_param_grid={'svc__C':np.linspace(200,300,100),
#                     # 'pca__n_components':[25],
#                    'svc__gamma':np.linspace(0.00007,0.00087,100)}
# #
# gs=GridSearchCV(model,param_grid=ensemble_param_grid,cv=5,scoring='accuracy',n_jobs=-1,verbose=1)
# # rs=RandomizedSearchCV(model,param_distributions=ksvm_param_grid,cv=5,scoring='accuracy',n_jobs=-1,verbose=3,n_iter=30)
# gs.fit(X_std,y)
# print(gs.best_params_)
# print(gs.best_score_)
# print(gs.cv_results_)


# # Получение результатов кросс-валидации
# results = gs.cv_results_
#
# # Сортировка результатов по значению метрики качества
# sorted_results_idx = np.argsort(results['mean_test_score'])[::-1]
#
# # Вывод первых 15 лучших классификаторов
# for i in range(50):
#     idx = sorted_results_idx[i]
#     print(f"Лучшие параметры для классификатора {i+1}:")
#     print(results['params'][idx])
#     print(f"Средняя оценка на кросс-валидации: {results['mean_test_score'][idx]}")



# #
# kFold = StratifiedKFold(n_splits=3,shuffle=True)
#
# scores=[]
# import tensorflow
# lrelu = lambda x: tensorflow.keras.activations.relu(x, alpha=0.1)
# for train, test in kFold.split(X_std, y):
#     model = Sequential()
#     model.add(Dense(64, input_shape=(X_std.shape[1],), activation=lrelu,kernel_initializer='he_uniform'))
#     model.add(Dropout(0.1))
#     model.add(Dense(20, activation=lrelu,kernel_initializer='he_uniform'))
#     model.add(Dropout(0.1))
#     model.add(Dense(10, activation=lrelu,kernel_initializer='he_uniform'))
#     model.add(Dropout(0.1))
#     model.add(Dense(1, activation='sigmoid',kernel_initializer='he_uniform'))
#     # optmzr = opt.Adam(learning_rate=0.003)
#     optmzr = opt.SGD(learning_rate=0.005,momentum=0.9,nesterov=True)
#     model.compile(loss='binary_crossentropy', optimizer=optmzr, metrics=['accuracy'])
#     model.fit(X_std[train], y[train], epochs=40, batch_size=32, verbose=1,use_multiprocessing=True)
#
#     res=model.evaluate(X_std[test], y[test])
#     scores.append(res)
#
#
# scores=np.array(scores)
# print(f' [loss,accuracy] {scores.mean(axis=0)} +- {scores.std(axis=0)}')

# model = Sequential()
# model.add(Dense(64, input_shape=(X_std.shape[1],), activation=lrelu, kernel_initializer='he_uniform'))
# model.add(Dropout(0.1))
# model.add(Dense(20, activation=lrelu, kernel_initializer='he_uniform'))
# model.add(Dropout(0.1))
# model.add(Dense(10, activation=lrelu, kernel_initializer='he_uniform'))
# model.add(Dropout(0.1))
# model.add(Dense(1, activation='sigmoid', kernel_initializer='he_uniform'))
# # optmzr = opt.Adam(learning_rate=0.003)
# optmzr = opt.SGD(learning_rate=0.005, momentum=0.9, nesterov=True)
# model.compile(loss='binary_crossentropy', optimizer=optmzr, metrics=['accuracy'])
# model.fit(X_std, y, epochs=50, batch_size=32, verbose=1, use_multiprocessing=True)
# pred=model.predict(ss.fit_transform(X2))#or fit transform????
# df2=pd.read_csv('spaceship-titanic/sample_submission.csv')
# df2['Transported']=pred.astype(bool)
# df2.to_csv('pred.csv',index=False)












