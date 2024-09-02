#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
import os

os.chdir('/home/ankitaC/Ankita/hsd-cnn/impact_scores/cifar100/vgg16/8')

# In[2]:


def filter_count(layer):
    if layer <= 3:
        return 64
    elif layer <= 10:
        return 128
    elif layer <= 20:
        return 256
    else:
        return 512
    


# In[3]:


def absolute(clas):
    layers = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]    
    b = np.zeros((len(layers)), dtype = np.float32)
    k = 0
    for i in layers:
        imp_vec = 'class'+str(clas)+'in'+str(i)+'.pkl'
        with open(imp_vec, 'rb') as f:
            a = pickle.load(f)
        b[k] = np.sum((np.array(a)))
        #b[k] = np.linalg.norm((np.array(a)))
        k += 1
    return b


# In[4]:


classes = np.arange(100, dtype = np.uint8)

vectors = []

for i in classes:
    temp = absolute(i)
    vectors.append(temp)
vectors = np.array(vectors)


# In[5]:


variance = np.zeros((vectors.shape[0], vectors.shape[1]-1), dtype = np.float32)
for i in range(vectors.shape[0]):
    for j in range(vectors.shape[1]-1):
        temp = [vectors[i][j]]
        temp.append(vectors[i][j+1])
        variance[i][j] = np.var(np.array(temp))


# In[6]:


temp = []
for i in range(variance.shape[0]):
    temp.append(np.argsort(variance[i])[-3:])
temp = np.array(temp)
#print(temp)
temp = temp.flatten()


# In[7]:


from collections import Counter

x = Counter(temp)
#x = np.array(x.most_common())
x = np.array(x.most_common())
final = []
for i in range(x.shape[0]):
    if x[i, 1] > 2:
        final.append(x[i, 0])


# In[8]:


final = np.array(final)
mask = np.zeros((13), dtype = np.uint8)
for i in final:
    mask[i+1] = 1
print(mask)

