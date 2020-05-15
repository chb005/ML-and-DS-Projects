#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("diabetes.csv")


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.isnull().values.any()


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[7]:


df.corr()


# In[8]:


diabetes_map =pd.get_dummies(df['diabetes'])
diabetes_map


# In[9]:


diabetes_map = {True: 1, False: 0}


# In[10]:


df['diabetes'] = df['diabetes'].map(diabetes_map)


# In[11]:


df.head(5)


# In[12]:


diabetes_true_count = len(df.loc[df['diabetes'] == True])
diabetes_false_count = len(df.loc[df['diabetes'] == False])


# In[13]:


(diabetes_true_count,diabetes_false_count)


# In[14]:


from sklearn.model_selection import train_test_split
feature_columns = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']
predicted_class = ['diabetes']


# In[15]:


X = df[feature_columns].values
y = df[predicted_class].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=10)


# In[16]:


print("total number of rows : {0}".format(len(df)))
print("number of rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc'] == 0])))
print("number of rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp'] == 0])))
print("number of rows missing insulin: {0}".format(len(df.loc[df['insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(df.loc[df['bmi'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(df.loc[df['diab_pred'] == 0])))
print("number of rows missing age: {0}".format(len(df.loc[df['age'] == 0])))
print("number of rows missing skin: {0}".format(len(df.loc[df['skin'] == 0])))


# In[17]:



from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

#fill_values = Imputer(missing_values=0, strategy="mean", axis=0)

X_train = imputer.fit_transform(X_train)
X_test = imputer.fit_transform(X_test)


# In[47]:


from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=5,n_estimators=5)

random_forest_model.fit(X_train, y_train.ravel())


# In[48]:


predict_train_data = random_forest_model.predict(X_test)

from sklearn import metrics

print("Accuracy = {0:.3f}".format(metrics.accuracy_score(y_test, predict_train_data)))


# In[46]:


params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}


# In[37]:


from sklearn.model_selection import RandomizedSearchCV
import xgboost


# In[38]:


classifier=xgboost.XGBClassifier()


# In[39]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)


# In[40]:


classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.3, gamma=0.0, learning_rate=0.25,
       max_delta_step=0, max_depth=3, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)


# In[41]:


from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,X,y.ravel(),cv=10)


# In[42]:


score


# In[43]:


score.mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




