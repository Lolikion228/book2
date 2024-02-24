import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


df=pd.read_csv('spaceship-titanic/train.csv')


# print(df.columns)

#feature 1
df=df.drop(columns=["PassengerId"])

#Feature 2
si2=SimpleImputer(missing_values=3,strategy='most_frequent')
le2=LabelEncoder()
df['HomePlanet']=le2.fit_transform(df['HomePlanet'])
df['HomePlanet']=si2.fit_transform(df['HomePlanet'].values.reshape(-1,1))

#Feature 3
si3=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df['CryoSleep']=df['CryoSleep'].map(int,na_action='ignore')
df['CryoSleep']=si3.fit_transform(df['CryoSleep'].values.reshape(-1,1))

#Feature 4

#Feature 5
si5=SimpleImputer(missing_values=3,strategy='most_frequent')
le5=LabelEncoder()
df['Destination']=le5.fit_transform(df['Destination'])
df['Destination']=si5.fit_transform(df['Destination'].values.reshape(-1,1))

#Feature 6
si6=SimpleImputer(missing_values=np.nan,strategy='mean')
df['Age']=si6.fit_transform(df['Age'].values.reshape(-1,1))

#Feature 7





