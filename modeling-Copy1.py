#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# In[7]:


dataset = pd.read_csv('baseline_lgb.csv')


# In[8]:


dataset.isna().sum()


# In[9]:


dataset['PredictedLogRevenue'].fillna(0, inplace = True)


# In[10]:


dataset.info()


# In[13]:


dataset['PredictedLogRevenue'] = dataset['PredictedLogRevenue'].astype(float)


# In[14]:


dataset['fullVisitorId'] = dataset['fullVisitorId'].astype(float)


# In[12]:


dataset.head()


# In[15]:


dataset.iloc[:,0:1]


# In[25]:


x = dataset.iloc[:,0:1]


# In[26]:


y = dataset.iloc[:,1]


# In[27]:


y.head()


# In[28]:


x.head()


# In[20]:


'''x = dataset.iloc[:, 0]
y = dataset.iloc[:, 1]'''


# In[24]:


x


# In[26]:


'''X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values'''


# In[2]:


get_ipython().system('pip install lightgbm')


# In[4]:


import lightgbm as lgbm


# In[35]:


from sklearn.linear_model import LinearRegression


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# In[31]:


regressor = lgbm.LGBMRegressor()


# In[32]:


regressor.fit(X_train, y_train)


# In[36]:


predictions = regressor.predict(X_test)


# In[37]:


predictions


# In[31]:


'''pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

print(model.predict([[18966950000000]]))'''


# In[39]:


import pickle
file = open('lgbm_model.pkl','wb')

pickle.dump(regressor,file)


# In[ ]:




