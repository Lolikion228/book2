import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold,GridSearchCV



df=pd.read_csv('spaceship-titanic/train.csv')

y=df['Transported'].map(int).values

df=df.drop(columns=['Transported'])
# print(df.columns)


#feature 1
df=df.drop(columns=["PassengerId"])


#Feature 2
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



#Feature 5
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



#Feature 4
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
ss=StandardScaler()
X_std=np.copy(X)
X_std=ss.fit_transform(X_std)



# lr=LogisticRegression(C=0.01,max_iter=10000,penalty='l2')
# lr.fit(X,y)
# print(lr.score(X,y))
# lr.fit(X_std,y)
# print(lr.score(X_std,y))

# ksvm=SVC(kernel='rbf',gamma=0.1,C=1,max_iter=10000)
# ksvm.fit(X,y)
# print(ksvm.score(X,y))
# ksvm.fit(X_std,y)
# print(ksvm.score(X_std,y))

# tree=DecisionTreeClassifier(criterion='entropy',max_depth=30)
# tree.fit(X,y)
# print(tree.score(X,y))
# tree.fit(X_std,y)
# print(tree.score(X_std,y))


