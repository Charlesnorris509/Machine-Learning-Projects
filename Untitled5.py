#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape

fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for target, image, ax in zip(people.target, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(people.target_names[target])


# In[11]:


print("people.images.shape: {}".format(people.images.shape))
print("Number of classes: {}".format(len(people.target_names)))


# In[12]:


import numpy as np


# In[13]:


counts = np.bincount(people.target)

for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='   ')
    if (i + 1) % 3 == 0:
        print()


# In[14]:


mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1
    
X_people = people.data[mask]
y_people = people.target[mask]

X_people = X_people / 255.


# In[15]:


import mglearn


# In[16]:


mglearn.plots.plot_pca_whitening()


# In[17]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X_people,y_people, stratify = y_people, random_state=0)

Knn = KNeighborsClassifier()
Knn.fit(X_train, Y_train)
Knn.score(X_test, Y_test)


# In[19]:


from sklearn.decomposition import PCA


# In[20]:


pca = PCA(n_components = 100, whiten = True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("X_train principal coomponent analysis shape :", X_train_pca.shape)


# In[21]:


Knn.fit(X_train_pca, Y_train)
Knn.score(X_test_pca, Y_test)


# In[23]:


print("PCA component shape :", pca.components_.shape)


# In[24]:


fig, axes = plt.subplots(3, 5, figsize=(15, 12),
                         subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape),
              cmap='viridis')
    ax.set_title("{}. component".format((i + 1)))


# In[25]:


mglearn.plots.plot_pca_faces(X_train ,X_test, image_shape)


# In[27]:


mglearn.discrete_scatter(X_train_pca[:,0], X_train_pca[:,1], Y_train)


# In[ ]:




