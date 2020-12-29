#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import sklearn


# In[93]:


train=pd.read_csv("C:/Users/HP/Downloads/7b9447c625cf11eb/dataset/train.csv")
test=pd.read_csv("C:/Users/HP/Downloads/7b9447c625cf11eb/dataset/test.csv")
train.head(20)


# In[94]:


train['instock_date'] = pd.to_datetime(train.instock_date, format='%Y-%m-%d %H:%M:%S')
train = train.assign(hour=train.instock_date.dt.hour,
               day=train.instock_date.dt.day,
               month=train.instock_date.dt.month,
               year=train.instock_date.dt.year)
train = train.drop(['instock_date'],axis = 1)
train.head()


# In[95]:


test['instock_date'] = pd.to_datetime(test.instock_date, format='%Y-%m-%d %H:%M:%S')
test = test.assign(hour=test.instock_date.dt.hour,
               day=test.instock_date.dt.day,
               month=test.instock_date.dt.month,
               year=test.instock_date.dt.year)
test.head()
test = test.drop(['instock_date'],axis = 1)


# In[96]:


print(train['Market_Category'].unique())
print(train['Stall_no'].unique())
print(train.index)
test.head()


# In[97]:


print(train['Selling_Price'].isnull().sum())
train['SellingPrice'] = train['Selling_Price'].fillna(train['Selling_Price'].median())
print(train['SellingPrice'].isnull().sum())
train.head()


# In[98]:


print(train['Selling_Price'].mode())
print(train['Selling_Price'].median())
print(train['Selling_Price'].mean())


# In[99]:


features=['Stall_no', 'Market_Category', 'Loyalty_customer', 'Product_Category', 'Grade', 'Demand', 'Discount_avail', 'charges_1', 'charges_2 (%)', 'Minimum_price', 'Maximum_price', 'hour', 'day', 'month', 'year']
X=train[features]
y=train.SellingPrice
X.head(30)


# In[100]:


cols_with_missing = [col for col in X.columns
                     if X[col].isnull().any()]
print(cols_with_missing)


# In[101]:


print(train['Discount_avail'].isnull().sum())
print(train['charges_1'].isnull().sum())
print(train['charges_2 (%)'].isnull().sum())
print(train['Minimum_price'].isnull().sum())
print(train['Maximum_price'].isnull().sum())


# In[102]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[103]:


X_test = test[features]
X_test.head()


# In[104]:


from sklearn.preprocessing import LabelEncoder

# Get list of categorical variables
s = (train_X.dtypes == 'object')
object_cols = list(s[s].index)

# Make copy to avoid changing original data 
label_X_train = train_X.copy()
label_X_test = X_test.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(train_X[col])
    label_X_test[col] = label_encoder.transform(X_test[col])


# In[105]:


from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(label_X_train))
imputed_X_test = pd.DataFrame(my_imputer.transform(label_X_test))

# Imputation removed column names; put them back
imputed_X_train.columns = label_X_train.columns
imputed_X_test.columns = label_X_test.columns

cols_with_missing2 = [col for col in imputed_X_train.columns
                     if imputed_X_train[col].isnull().any()]
print(cols_with_missing2)

cols_with_missing3 = [col for col in train.columns
                     if train[col].isnull().any()]
print(cols_with_missing3)


# In[106]:


from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LinearRegression
model=RandomForestRegressor(random_state=1)
model.fit(imputed_X_train,train_y)
preds=model.predict(imputed_X_test)
print(preds) 
for i in range(len(preds)):
    preds[i]=abs(preds[i])
    
print(preds)    


# In[107]:


df = pd.DataFrame({'Product_id':list(test['Product_id']),
                    'Selling_Price':(list(preds))})
submission_data=df
submission_data.to_csv('Selling_Price4.csv', index=False)


# In[ ]:




