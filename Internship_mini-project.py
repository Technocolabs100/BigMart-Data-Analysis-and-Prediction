#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dtale
import klib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[3]:


df = pd.read_csv(r"C:\Users\yahya\OneDrive\Bureau\Data\Train.csv")


# In[4]:


df.head()


# In[5]:


df


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df["Item_Weight"].fillna(df["Item_Weight"].mean(), inplace = True)


# In[9]:


df.isnull().sum()


# In[10]:


df["Outlet_Size"].value_counts()


# In[11]:


df["Outlet_Size"].mode()


# In[12]:


df["Outlet_Size"].fillna(df["Outlet_Size"].mode()[0], inplace = True)


# In[13]:


df.isnull().sum()


# In[14]:


df


# In[15]:


df.drop(['Item_Identifier','Outlet_Identifier'],axis=1, inplace= True)


# In[16]:


df


# In[20]:


dtale.show(df)


# In[29]:


df = klib.data_cleaning(df)


# In[23]:


df=klib.convert_datatypes(df)


# In[33]:


le=LabelEncoder()
df['item_fat_content']= le.fit_transform(df['item_fat_content'])
df['item_type']= le.fit_transform(df['item_type'])
df['outlet_size']= le.fit_transform(df['outlet_size'])
df['outlet_location_type']= le.fit_transform(df['outlet_location_type'])
df['outlet_type']= le.fit_transform(df['outlet_type'])


# In[34]:


X=df.drop('item_outlet_sales',axis=1)
Y=df['item_outlet_sales']


# In[36]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=101, test_size=0.2)


# In[37]:


X.describe()


# In[39]:


sc= StandardScaler()
X_train_std= sc.fit_transform(X_train)
X_test_std= sc.transform(X_test)
X_train_std


# In[41]:


X_test_std


# In[42]:


Y_train


# In[43]:


Y_test


# In[45]:


lr= LinearRegression()
lr.fit(X_train_std,Y_train)


# In[46]:


X_test.head()


# In[48]:


Y_pred_lr=lr.predict(X_test_std)


# In[49]:


print(r2_score(Y_test,Y_pred_lr))
print(mean_absolute_error(Y_test,Y_pred_lr))
print(np.sqrt(mean_squared_error(Y_test,Y_pred_lr)))


# In[ ]:
