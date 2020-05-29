#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[3]:


df=pd.read_csv('Data.csv', encoding = "ISO-8859-1")


# In[6]:


df.head()


# In[7]:


train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']


# In[8]:


data=train.iloc[:,2:27]
data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

# Renaming column names for ease of access
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
data.head(5)


# In[11]:


for index in new_Index:
    data[index]=data[index].str.lower()
data.head(2)


# In[10]:


' '.join(str(x) for x in data.iloc[1,0:25])


# In[12]:


headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))


# In[13]:


headlines[0]


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


# In[15]:


countvector=CountVectorizer(ngram_range=(2,2))
traindataset=countvector.fit_transform(headlines)


# In[16]:


randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])


# In[20]:


test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)


# In[21]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[23]:


matrix=confusion_matrix(test['Label'],predictions)
print(matrix)


# In[26]:


randomclassifier.score


# In[35]:


precision=(matrix[0][0]/(matrix[0][0]+matrix[1][0]))*100


# In[36]:


precision


# In[40]:


Accuracy=(matrix[0][0]+matrix[1][1])/((matrix[0][0]+matrix[0][1]+matrix[1][0]+matrix[1][1]))*100


# In[41]:


Accuracy


# In[ ]:




