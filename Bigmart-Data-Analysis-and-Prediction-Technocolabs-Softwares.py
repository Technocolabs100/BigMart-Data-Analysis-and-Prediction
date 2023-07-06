#!/usr/bin/env python
# coding: utf-8

# 
# # Probelm Statement

# The aim of this data science project is to build a predictive model and find out the sales of each product at a particular store. By exploring the properties of the products and the stores.

# # Hypothesis generation 

# BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, there are two factors in this dataset.First one is Product features and second one is Store features.

# # Importing libraries and reading data

# Feature Engineering and Preprocessing

# In[1]:


import pandas as pd
import scipy
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# Data Visualization

# In[57]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pylab
from pylab import *
from pandas_profiling import ProfileReport


# Modelling

# In[103]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import xgboost as xg
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# Connecting Dataset

# In[4]:


df_train= pd.read_csv(r"C:\Users\nikhi\Downloads\9961_14084_bundle_archive\Train.csv")
df_test= pd.read_csv(r"C:\Users\nikhi\Downloads\9961_14084_bundle_archive\Test.csv")


# In[5]:


df_train.shape


# In[6]:


df_test.shape


# In[7]:


df_train.head()


# # Data Preprocessing

# In[8]:


df_train.info()


# In[9]:


plt.figure(figsize = (16, 6))
plt.title("Distribution of weight")
sns.kdeplot(data = df_train['Item_Weight'], fill = True)


# The graph shows that the weight is normally distributed (no extreme outliers) so we can replace the nulls with mean(because the mean is affected by outliers).

# In[10]:


mean_value=df_train['Item_Weight'].mean()
df_train['Item_Weight'].fillna(value=mean_value, inplace=True)


# In[11]:


mode_value=df_train.Outlet_Size.mode()[0]
df_train['Outlet_Size'].fillna(value=mode_value, inplace=True)


# In[12]:


df_train.info()


# In[13]:


print(df_train['Item_Fat_Content'].unique())
print(df_train['Item_Type'].unique())
print(df_train['Outlet_Size'].unique())
print(df_train['Outlet_Location_Type'].unique())
print(df_train['Outlet_Type'].unique())


# In[14]:


df_train['Item_Fat_Content'] = df_train['Item_Fat_Content'].replace(['LF'], 'low fat')
df_train['Item_Fat_Content'] = df_train['Item_Fat_Content'].replace(['Low Fat'], 'low fat')
df_train['Item_Fat_Content'] = df_train['Item_Fat_Content'].replace(['reg'], 'Regular')


# In[15]:


print(df_train['Item_Fat_Content'].unique())


# Checking Duplicates Values

# In[16]:


df_train.duplicated().sum()


# In[17]:


df_train.describe().T


# # Exploratory Data Analysis

# In[18]:


df_train.plot(
    kind='box',
    subplots=True,
    sharey=False,
    figsize=(10, 6)
)

# increase spacing between subplots
plt.subplots_adjust(wspace=0.5)
plt.show()


# In[19]:


def outliers(dataset,col):
  Q1 = dataset[col].quantile(0.25)
  Q3 = dataset[col].quantile(0.75)
  IQR = Q3-Q1
  lower_bound = Q1-1.5*IQR
  upper_bound = Q3+1.5*IQR

  for i in range(len(dataset)):
      if dataset[col].iloc[i] > upper_bound:
          dataset[col].iloc[i] = upper_bound
      if dataset[col].iloc[i] < lower_bound:
          dataset[col].iloc[i] = lower_bound
            
outliers(df_train,'Item_Visibility')
outliers(df_train,'Item_Outlet_Sales')


# In[20]:


df_train.plot(
    kind='box',
    subplots=True,
    sharey=False,
    figsize=(10, 6)
)

# increase spacing between subplots
plt.subplots_adjust(wspace=0.5)
plt.show()


# In[21]:


corr = df_train.corr()
plt.figure(dpi=130)
sns.heatmap(df_train.corr(), annot=True, fmt= '.2f')
plt.show()


# # Univariate Analysis

# In[22]:


plt.hist(df_train['Item_Fat_Content'])


# In[23]:


plt.figure(figsize=(45,18))
plt.hist(df_train['Item_Type'], edgecolor='black', bins=16)


# In[24]:


plt.hist(df_train['Item_Weight'], edgecolor='black')


# It is obvious that the majority products in the data have weight between 10 and 15(medium).

# In[25]:


plt.hist(df_train['Item_Visibility'], edgecolor='black')


# Items with lower visibility are more than that with the higher visibility, this indicates that in order to increase the item visibility, the store must provide more money, to make the item more visible (for example, by ads), so that's why items with higher visibility are few in the data

# In[26]:


plt.hist(df_train['Outlet_Size'], edgecolor='black')


# The previous graph indicates that medium is the common size in stores.

# In[27]:


plt.figure(figsize=(15,8))
plt.hist(df_train['Outlet_Type'], edgecolor='black', bins = 4)


# # Bivariate Analysis

# In[28]:


plt.scatter(df_train.Item_MRP, df_train.Item_Outlet_Sales)
plt.title('item MRP vs. sales')
plt.xlabel('item MRP')
plt.ylabel('sales')


# The previous graph assured that there is a strong relationship between item MRP and outlet_item sales!

# In[29]:


sns.barplot(x = 'Outlet_Establishment_Year',y = 'Item_Outlet_Sales',data = df_train)
plt.show()


# That bar plot shows that lowest sales are in the outlets established between 1995 and 2000.

# In[30]:


sns.set(rc={'figure.figsize':(25,20)})
sns.barplot(x = 'Item_Type',y = 'Item_Outlet_Sales',data = df_train)

plt.show()


# This shows that starchy foods make highest sales followed by seafood, although both of them are from the least found items in the data, so this means they make very high sales but not due to their frequency.

# In[31]:


sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x = 'Item_Fat_Content',y = 'Item_Outlet_Sales',data = df_train)
plt.show()


# In[32]:


sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x = 'Outlet_Size',y = 'Item_Outlet_Sales',data = df_train)
plt.show()


# The bar plot shows that sales are higher in high and medium outlets and lower in small outlets.

# In[33]:


sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x = 'Outlet_Location_Type',y = 'Item_Outlet_Sales',data = df_train)
plt.show()


# The previous plot shows that highest sales are in Tier 2 cities(medium developed markets)

# In[34]:


sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x = 'Outlet_Type',y = 'Item_Outlet_Sales',data = df_train)
plt.show()


# The previous plot shows that supermarket type 3 has reached the highest sales upon the outlet types, and that grocery store has the least sales. so we can conclude that the grocery store not as important as the supermarket.

# # EDA Using Pandas

# In[58]:


profile = ProfileReport(df_train, title="Pandas Profiling Report")


# In[59]:


profile


# # Feature Engineering

# In[35]:


df_train.drop(['Item_Identifier', 'Outlet_Identifier','Item_Weight'], axis=1, inplace = True)


# In[36]:


df_train.head(10)


# # Encoding

# Label Encoding

# In[37]:


df_train = df_train.apply(le.fit_transform)
df_test = df_test.apply(le.fit_transform)


# In[38]:


df_train


# In[41]:


df_test


# # Separate Data's

# In[67]:


y=df_train["Item_Outlet_Sales"]
x=df_train.drop("Item_Outlet_Sales", axis=1)


# In[68]:


x_train, x_test, y_train, y_test = train_test_split(x, y)


# In[69]:


x.describe()


# In[71]:


x_train_std= sc.fit_transform(x_train)


# In[72]:


x_test_std= sc.transform(x_test)


# In[73]:


x_train_std


# In[74]:


x_test_std


# In[75]:


y_train


# In[76]:


y_test


# # Hyper Parameter Tuning

# In[94]:


model = RandomForestRegressor()
n_estimators = [10, 100, 1000]
max_depth=range(1,31)
min_samples_leaf=np.linspace(0.1, 1.0)
max_features=["auto", "sqrt", "log2"]
min_samples_split=np.linspace(0.1, 1.0, 10)

# define grid search
grid = dict(n_estimators=n_estimators, max_depth=max_depth,max_features=max_features, )

#cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=101)

grid_search_forest = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, 
                           scoring='r2',error_score=0,verbose=2,cv=2)

grid_search_forest.fit(x_train_std, y_train)

# summarize results
print(f"Best: {grid_search_forest.best_score_:.3f} using {grid_search_forest.best_params_}")
means = grid_search_forest.cv_results_['mean_test_score']
stds = grid_search_forest.cv_results_['std_test_score']
params = grid_search_forest.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print(f"{mean:.3f} ({stdev:.3f}) with: {param}")


# In[ ]:





# In[99]:


grid_search_forest.best_params_


# In[100]:


grid_search_forest.best_score_


# In[101]:


Y_pred_rf_grid=grid_search_forest.predict(x_test_std)


# In[107]:


r2_score(y_test,Y_pred_rf_grid)


# # Modelling

# Linear Regression

# In[54]:


linear_model = LinearRegression().fit(x_train,y_train)
r_sq = linear_model.score(x_test, y_test)
predictions = linear_model.predict(x_test)
print("Mean Absolute Error: " + str(MAE(predictions, y_test)))
print(f"coefficient of determination: {r_sq}")


# Random Forest

# In[55]:


forest_model = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4,random_state=101)
forest_model.fit(x_train, y_train)
melb_preds = forest_model.predict(x_test)
print(MAE(y_test, melb_preds))
r_sq = forest_model.score(x_test, y_test)
print(f"coefficient of determination: {r_sq}")


# # XGboost

# In[56]:


my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, random_state = 0)
my_model.fit(x_train, y_train, early_stopping_rounds=5, eval_set=[(x_test, y_test)],verbose=False)
predictions = my_model.predict(x_test)
print("Mean Absolute Error: " + str(MAE(predictions, y_test)))
r_sq = my_model.score(x_test, y_test)
print(f"coefficient of determination: {r_sq}")


# # Summary

# We've done preprocessing, analysis, feature engineering and modeling on bigMart data whick includes some features of stores and items, to predict the sales of specific store for a specific item.We used machine learning models as linear regression, random forest and XGboost. The mean squared error of each model:-
# 
# Linear Regression -> 0.12
# 
# Random forset -> 0.10
# 
# XGboost -> 0.11

# In[ ]:




