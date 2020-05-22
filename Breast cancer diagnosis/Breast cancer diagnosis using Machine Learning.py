#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('cancer.csv')


# In[3]:


df.head()


# In[4]:


df.replace('?',-99999,inplace=True)


# In[5]:


df.drop(['id'],1,inplace=True)


# In[6]:


df.head()


# In[7]:


X=np.array(df.drop(['classes'],1))
y=np.array(df['classes'])


# In[8]:


X


# In[9]:


y


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[11]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[12]:


X_train


# In[13]:


X_test


# In[14]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)
explained_variance=pca.explained_variance_ratio_


# In[15]:


X_train


# In[16]:


X_test


# In[17]:


from sklearn.neighbors import KNeighborsClassifier
knn = []
for i in range(1,21):
           
    classifier = KNeighborsClassifier(n_neighbors=i)
    trained_model=classifier.fit(X_train,y_train)
    trained_model.fit(X_train,y_train )


# In[18]:


y_pred = classifier.predict(X_test)


# In[19]:


from sklearn.metrics import confusion_matrix


# In[20]:


cm_KNN = confusion_matrix(y_test, y_pred)
print(cm_KNN)
print("Accuracy score of train KNN")
print(accuracy_score(y_train, trained_model.predict(X_train))*100)
    
print("Accuracy score of test KNN")
print(accuracy_score(y_test, y_pred)*100)
    
knn.append(accuracy_score(y_test, y_pred)*100)


# In[21]:


from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)

trained_model=classifier.fit(X_train,y_train)
trained_model.fit(X_train,y_train )


# In[22]:


y_pred = classifier.predict(X_test)


# In[23]:


from sklearn.metrics import confusion_matrix
cm_SVM = confusion_matrix(y_test, y_pred)
print(cm_SVM)
print("Accuracy score of train SVM")
print(accuracy_score(y_train, trained_model.predict(X_train))*100)

print("Accuracy score of test SVM")
print(accuracy_score(y_test, y_pred)*100)


# In[ ]:




