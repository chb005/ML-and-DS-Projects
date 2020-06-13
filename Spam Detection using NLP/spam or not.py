#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])


# In[4]:


messages['label']


# In[5]:


messages['message']


# In[7]:


#Data cleaning and preprocessing
import re
import nltk
nltk.download('stopwords')


# In[8]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[9]:


ps = PorterStemmer()
corpus = []


# In[10]:


for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)


# In[14]:


review


# In[15]:


corpus


# In[16]:


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()


# In[17]:


y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# In[18]:


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[19]:


# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)


# In[20]:


from sklearn.metrics import confusion_matrix
confun_m=confusion_matrix(y_test,y_pred)


# In[21]:


confun_m


# In[22]:


from sklearn.metrics import accuracy_score
accu_s=accuracy_score(y_test,y_pred)


# In[23]:


accu_s


# In[ ]:




