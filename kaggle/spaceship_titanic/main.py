import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


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
print(df)