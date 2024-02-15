from PIL import Image
import numpy as np
from numpy import asarray
from sklearn.model_selection import train_test_split
import os
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



def compute_ch_mean(ch_copy):
    sum_hairs = ch_copy[0:30, 0:96].sum() + ch_copy[0:60, 0:20].sum() + ch_copy[0:60, 75:96].sum()
    sum_face = ch_copy[20:60, 20:75].sum()#25 x
    sum_hairs /= (30 * 96) + (60 * 20) + (60 * 20)
    sum_face /= 41 * 56
    return np.array([sum_face, sum_hairs, (sum_face**2+sum_hairs**2)])

def create_features(img,p=3,bias=0.1):
    channels =[img[:, :, i] for i in range(3)]
    ch_means=np.array([compute_ch_mean(ch) for ch in channels])

    ch_means=np.concatenate(ch_means,axis=0)
    ch_means=list(ch_means)
    for j in range(3):
        ch_means.append(channels[j].mean())

    return np.log(bias+np.array(ch_means)**p)


X=[]
y=[]
classes=os.listdir("dataset")
for ix,class_ in enumerate(classes):
    pics=os.listdir(f'dataset/{class_}')
    for pic in pics:
        img=asarray(Image.open(f'dataset/{class_}/{pic}'))
        X.append(np.array(img))
        y.append(ix)


# X=np.array(X)
# y=np.array(y)

for i in range(len(X)):
    X[i]=create_features(X[i])

X=np.array(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)


X_test_std=(X_test-X_train.mean())/X_train.std()
X_train_std=(X_train-X_train.mean())/X_train.std()


ppn=Perceptron(eta0=0.1,l1_ratio=0,alpha=0.1)
lr=LogisticRegression(solver='lbfgs',C=1,multi_class='ovr')
Svc

classifier=lr
classifier.fit(X_train_std,y_train)
print(str(classifier))
print(classifier.score(X_test_std,y_test))
# print(set(classifier.predict(X_test)))