import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from numpy import asarray





cl_0=np.array([ asarray(Image.open(f'pics/0_{i}.jpg')) for i in range(5)  ])
cl_1=[ asarray(Image.open(f'pics/1_{i}.jpg')) for i in range(5)  ]
cl_2=[ asarray(Image.open(f'pics/2_{i}.jpg'))   for i in range(5)  ]

cl_0_std=np.copy(cl_0)
cl_0_std=(cl_0_std-cl_0_std.mean(axis=(0,1,2)))/cl_0_std.std(axis=(0,1,2))
cl_1_std=np.copy(cl_1)
cl_1_std=(cl_1_std-cl_1_std.mean(axis=(0,1,2)))/cl_1_std.std(axis=(0,1,2))
cl_2_std=np.copy(cl_2)
cl_2_std=(cl_2_std-cl_2_std.mean(axis=(0,1,2)))/cl_2_std.std(axis=(0,1,2))



# ex1=cl_0[0]
# ch2=np.copy(ex1[:,:,2])
# # ch2[0:60, 0:20]=0
# # ch2[0:60, 75:96]=0
# # ch2[0:30, 0:96]=0
# ch2[25:70, 20:75]=0
# plt.imshow(ch2)
# plt.show()
# plt.close()

def compute_ch_mean(ch_copy):
    sum_hairs = ch_copy[0:30, 0:96].sum() + ch_copy[0:60, 0:20].sum() + ch_copy[0:60, 75:96].sum()
    sum_face = ch_copy[20:60, 20:75].sum()#25 x
    sum_hairs /= (30 * 96) + (60 * 20) + (60 * 20)
    sum_face /= 41 * 56
    return np.array([sum_face, sum_hairs, (sum_face**2+sum_hairs**2)])

def create_features(img,p=3,bias=0.1):
    channels = [img[:, :, i] for i in range(3)]
    ch_means=[compute_ch_mean(ch) for ch in channels]
    feature_vector=np.array([x[0] for x in ch_means])
    # feature_vector=np.sum(ch_means, axis=0)
    return np.log(bias+feature_vector**p)


def minkowski(v1,v2,p=2):
    v=v1-v2
    return np.sum(np.abs(v)**p)**(1/p)


cl_0_feats=[create_features(img) for img in cl_0_std]
cl_1_feats=[create_features(img) for img in cl_1_std]
cl_2_feats=[create_features(img) for img in cl_2_std]





x0=[ex[0] for ex in cl_0_feats]
y0=[ex[1] for ex in cl_0_feats]
z0=[ex[2] for ex in cl_0_feats]

x1=[ex[0] for ex in cl_1_feats]
y1=[ex[1] for ex in cl_1_feats]
z1=[ex[2] for ex in cl_1_feats]


x2=[ex[0] for ex in cl_2_feats]
y2=[ex[1] for ex in cl_2_feats]
z2=[ex[2] for ex in cl_2_feats]




fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.scatter(x0,y0,z0,c='green')
ax.scatter(x1,y1,z1,c='red')
ax.scatter(x2,y2,z2,c='blue')
plt.title('feat first')
plt.show()
plt.close()



