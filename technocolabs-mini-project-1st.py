#!/usr/bin/env python
# coding: utf-8

# # import packages

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# # load data

# In[2]:


train_data_path='D://Technocolabs-ML-internship//first-mini-project//9961_14084_bundle_archive//Train.csv'
test_data_path='D://Technocolabs-ML-internship//first-mini-project//9961_14084_bundle_archive//Test.csv'
train_data=pd.read_csv(train_data_path)
test_data=pd.read_csv(test_data_path)


# In[3]:


train_data 


# In[4]:


test_data


# # Exploratory Data Analysis
# 
# 

# In[5]:


train_data.shape , test_data.shape


# In[6]:


train_data.columns , test_data.columns


# In[7]:


categorial_features = train_data.select_dtypes(include=[np.object])
categorial_features.head(2)


# In[8]:


categorial_features = test_data.select_dtypes(include=[np.object])
categorial_features.head(2)


# In[9]:


numerical_features = train_data.select_dtypes(include=[np.number])
numerical_features.head(2)


# In[10]:


numerical_features = test_data.select_dtypes(include=[np.number])
numerical_features.head(2)


# # checking missing values

# In[11]:


train_data.describe()


# In[12]:


train_missing_values_count=train_data.isnull().sum()
test_missing_values_count=test_data.isnull().sum()

train_missing_values_count , test_missing_values_count


# In[13]:


train_total_cells=np.product(train_data.shape)
test_total_cells=np.product(test_data.shape)
train_total_missing=train_missing_values_count.sum()
test_total_missing=test_missing_values_count.sum()
train_percentage_missing=(train_total_missing/train_total_cells)*100
test_percentage_missing=(test_total_missing/test_total_cells)*100


# In[14]:


print('percentage of  training cells : ',train_percentage_missing ,'%')
print('percentage of  testing cells : ',test_percentage_missing ,'%')


# # visualizations

# In[15]:


import seaborn as sns


# In[16]:


train_data.Outlet_Establishment_Year.unique()


# In[17]:


# plt.figure(figsize=(6,6))
# train_data.hist(column='Outlet_Establishment_Year', grid=False, edgecolor='black')
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=train_data)
plt.show()


# In[18]:


plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=train_data)
plt.show()


# In[19]:


sns.kdeplot(train_data['Item_MRP'])


# In[20]:


sns.distplot(train_data['Item_Outlet_Sales'])


# In[21]:


plt.hist(train_data.Item_Fat_Content)
# plt.hist(train_data.Item_Fat_Content)


# In[22]:


plt.scatter(train_data.Item_Visibility, train_data.Item_Type)


# In[23]:


train_data.corr()


# # checking outliers

# In[24]:


train_data.boxplot(column=['Item_Weight'])
plt.show()


# In[25]:


mean_Item_Weight=train_data['Item_Weight'].mean()
train_data['Item_Weight'].replace(np.nan,mean_Item_Weight,inplace=True)


# In[26]:


train_data.boxplot(column=['Item_Visibility'])
plt.show()


# In[27]:


median_Item_Visibility=train_data['Item_Visibility'].median()
train_data['Item_Visibility'].replace(np.nan,median_Item_Visibility,inplace=True)
train_data['Item_Visibility'].replace(0,median_Item_Visibility,inplace=True)


# In[28]:


train_data.boxplot(column=['Item_MRP'])
plt.show()


# In[29]:


train_data.boxplot(column=['Outlet_Establishment_Year'])
plt.show()


# In[30]:


train_data.boxplot(column=['Item_Outlet_Sales'])
plt.show()


# In[31]:


median_Item_Outlet_Sales=train_data['Item_Outlet_Sales'].median()
train_data['Item_Outlet_Sales'].replace(np.nan,median_Item_Outlet_Sales,inplace=True)


# In[32]:


mode_Outlet_Size=train_data['Outlet_Size'].mode().values[0]
train_data['Outlet_Size']=train_data['Outlet_Size'].replace(np.nan,mode_Outlet_Size)


# In[33]:


train_missing_values_count=train_data.isnull().sum()
train_missing_values_count


# # check duplicate

# In[34]:


duplicate= train_data.duplicated()
print(duplicate.sum())
train_data[duplicate]


# # handling outliers

# In[35]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range=Q1-(1.5*IQR)
    upper_range=Q3+(1.5*IQR)
    return lower_range , upper_range


# In[36]:


lowincome,uppincome=remove_outlier(train_data['Item_Visibility'])
train_data['Item_Visibility']=np.where(train_data['Item_Visibility']>uppincome,uppincome,train_data['Item_Visibility'])
train_data['Item_Visibility']=np.where(train_data['Item_Visibility']<lowincome,lowincome,train_data['Item_Visibility'])


# In[37]:


train_data.boxplot(column=['Item_Visibility'])
plt.show()


# In[38]:


lowincome,uppincome=remove_outlier(train_data['Item_Outlet_Sales'])
train_data['Item_Outlet_Sales']=np.where(train_data['Item_Outlet_Sales']>uppincome,uppincome,train_data['Item_Outlet_Sales'])
train_data['Item_Outlet_Sales']=np.where(train_data['Item_Outlet_Sales']<lowincome,lowincome,train_data['Item_Outlet_Sales'])


# In[39]:


train_data.boxplot(column=['Item_Outlet_Sales'])
plt.show()


# In[40]:


train_missing_values_count=train_data.isnull().sum()
train_missing_values_count


# In[41]:


train_data.describe()


# In[42]:


from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
# define min max scaler
scaler = MinMaxScaler()
# transform data
# train_data['Item_Weight'] = scaler.fit_transform(train_data[['Item_Weight']])
# train_data['Item_Visibility'] = scaler.fit_transform(train_data[['Item_Visibility']])
train_data['Item_MRP'] = scaler.fit_transform(train_data[['Item_MRP']])
# train_data['Item_Outlet_Sales'] = scaler.fit_transform(train_data[['Item_Outlet_Sales']])
# train_data['Outlet_Establishment_Year'] = scaler.fit_transform(train_data[['Outlet_Establishment_Year']])


# In[43]:


train_data.describe()


# In[44]:


import seaborn as sns


# In[45]:


plt.hist(train_data.Item_Fat_Content)
# plt.hist(train_data.Item_Fat_Content)


# In[46]:


categorial_features = train_data.select_dtypes(include=[np.object])
categorial_features.head(1)


# In[47]:


train_data.Item_Identifier.value_counts() ,train_data.Item_Identifier.value_counts().describe()


# In[48]:


train_data.Item_Fat_Content.value_counts()


# In[49]:


train_data['Item_Fat_Content']=train_data['Item_Fat_Content'].replace('low fat','Low Fat')
train_data['Item_Fat_Content']=train_data['Item_Fat_Content'].replace('LF','Low Fat')
train_data['Item_Fat_Content']=train_data['Item_Fat_Content'].replace('reg','Regular')


# In[50]:


train_data['Item_Fat_Content'].unique()


# In[51]:


train_data.Item_Fat_Content.value_counts()


# In[52]:


train_data.Item_Type.value_counts() , train_data.Item_Type.value_counts().describe()


# In[53]:


train_data.Outlet_Identifier


# In[54]:


train_data.Outlet_Identifier.value_counts() , train_data.Item_Type.value_counts().describe()


# In[55]:


train_data.Outlet_Identifier.unique()


# In[56]:


train_data.Outlet_Size.value_counts()


# In[57]:


train_data.Outlet_Location_Type.value_counts()


# In[58]:


train_data.Outlet_Type.value_counts()


# #  Encoding

# In[77]:


count_map_Item_Identifier = train_data['Item_Identifier'].value_counts().to_dict()
train_data['Item_Identifier'] = train_data['Item_Identifier'].map(count_map_Item_Identifier)


# In[78]:


count_map_Item_Identifier


# In[79]:


train_data['Item_Fat_Content'].unique()


# In[80]:


train_data['Item_Fat_Content']=pd.get_dummies(train_data['Item_Fat_Content'],drop_first=True)
# train_data['Item_Fat_Content']=pd.get_dummies(train_data['Item_Fat_Content'], drop_first=True).head()


# In[81]:


train_data['Item_Fat_Content'].unique()


# In[82]:


item_types=[]
for item in train_data['Item_Type'].unique():
    item_types.append(item)
    


# In[83]:


train_data['Item_Type']


# In[84]:


item_types


# In[85]:


Outlet_Identifier=[]
for OutletID in train_data['Outlet_Identifier'].unique():
    Outlet_Identifier.append(OutletID)


# In[86]:


Outlet_Identifier


# In[87]:


# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# le = preprocessing.LabelEncoder()
# le.fit(item_types)
# train_data['Item_Type']=le.transform(train_data['Item_Type'])
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(item_types)
train_data['Item_Type']=le.transform(train_data['Item_Type'])
# # >>>array([2, 2, 1]...)
# list(le.inverse_transform([2, 2, 1]))
# >>>['tokyo', 'tokyo', 'paris']


# In[88]:


le2 = preprocessing.LabelEncoder()
le2.fit(Outlet_Identifier)
train_data['Outlet_Identifier']=le2.transform(train_data['Outlet_Identifier'])


# In[89]:


Outlet_Size=[]
for Outletsize in train_data['Outlet_Size'].unique():
    Outlet_Size.append(Outletsize)


# In[90]:


Outlet_Size


# In[91]:


le3= preprocessing.LabelEncoder()
le3.fit(Outlet_Size)
train_data['Outlet_Size']=le3.transform(train_data['Outlet_Size'])


# In[92]:


Outlet_Type=[]
for Outlet_type in train_data['Outlet_Type'].unique():
    Outlet_Type.append(Outlet_type)


# In[93]:


Outlet_Type


# In[94]:


le4= preprocessing.LabelEncoder()
le4.fit(Outlet_Type)
train_data['Outlet_Type']=le4.transform(train_data['Outlet_Type'])


# In[95]:


Outlet_Location_Type=[]
for Outlet_type_loc in train_data['Outlet_Location_Type'].unique():
    Outlet_Location_Type.append(Outlet_type_loc)


# In[96]:


Outlet_Location_Type


# In[97]:


le5= preprocessing.LabelEncoder()
le5.fit(Outlet_Location_Type)
train_data['Outlet_Location_Type']=le5.transform(train_data['Outlet_Location_Type'])


# In[98]:


train_data


# # Splitting the dataset into the training and validation sets

# In[99]:


data=train_data
y=data.Item_Outlet_Sales
features=data.columns

X=data[features.drop('Item_Outlet_Sales')]


# In[100]:


X


# In[101]:


y


# In[102]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[103]:


from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score
forest_model = RandomForestRegressor(random_state=1)

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(forest_model,train_X, train_y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)


# In[104]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
mse=mean_absolute_error(val_y, melb_preds)
print(mse)


# In[105]:


# make predictions and calculate the MSE and y_variance


y_variance = np.var(melb_preds, ddof=1)

# calculate the accuracy percentage
accuracy = 100 * (1 - (mse / y_variance))

# print the accuracy percentage
print("Accuracy: {:.2f}%".format(accuracy))


# In[106]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(forest_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())


# In[107]:


from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=9)
my_model.fit(train_X, train_y)
predictions = my_model.predict(val_X)
mse=mean_absolute_error(val_y,predictions)
print("Mean Absolute Error: " + str(mse))


# In[108]:


# make predictions and calculate the MSE and y_variance


y_variance = np.var(predictions, ddof=1)

# calculate the accuracy percentage
accuracy = 100 * (1 - (mse / y_variance))

# print the accuracy percentage
print("Accuracy: {:.2f}%".format(accuracy))


# In[109]:


perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())


# In[110]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(train_X, train_y)
predictions = model.predict(val_X)
mse=mean_absolute_error(val_y,predictions)
print("Mean Absolute Error: " + str(mse))


# In[111]:


# make predictions and calculate the MSE and y_variance


y_variance = np.var(predictions, ddof=1)

# calculate the accuracy percentage
accuracy = 100 * (1 - (mse / y_variance))

# print the accuracy percentage
print("Accuracy: {:.2f}%".format(accuracy))


# In[112]:


from sklearn.linear_model import Ridge, Lasso, ElasticNet


#Ridge regularization
# ridge = Ridge() 
# ridge.fit(train_X, train_y)
# predictions = model.predict(val_X)
# print("Mean Absolute Error: " + str(mean_absolute_error(predictions, val_y)))
#Lasso regularization
# lasso = Lasso( alpha =0.25, tol = 0.0925)
# lasso.fit(train_X, train_y)
# predictions = model.predict(val_X)
# print("Mean Absolute Error: " + str(mean_absolute_error(predictions, val_y)))
#Elastic Net
elasticnet = ElasticNet()
elasticnet.fit(train_X, train_y)
predictions = model.predict(val_X)
mse=mean_absolute_error(val_y,predictions)
print("Mean Absolute Error: " + str(mse))


# In[113]:


# make predictions and calculate the MSE and y_variance


y_variance = np.var(predictions, ddof=1)

# calculate the accuracy percentage
accuracy = 100 * (1 - (mse / y_variance))

# print the accuracy percentage
print("Accuracy: {:.2f}%".format(accuracy))

