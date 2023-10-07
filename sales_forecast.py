#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# In[5]:


product_data = pd.read_csv('train.csv')
product_data.head(5)


# In[6]:


#cek null data
product_data.info()


# In[4]:


product_data = product_data.drop(['store', 'item'], axis=1)


# In[5]:


product_data.info()


# In[6]:


#convert date data type from object to datetime
product_data['date'] = pd.to_datetime(product_data['date'])


# In[7]:


product_data.info()


# In[8]:


#converting date to a month period and then sum the number of item in each month
product_data['date'] = product_data['date'].dt.to_period("M")
monthly_sales = product_data.groupby('date').sum().reset_index()


# In[9]:


# #convert the resulting date column to timestamp datatype
monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()


# In[12]:


monthly_sales.head(10)


# In[13]:


# Visualization
plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'], monthly_sales['sales'])
plt.xlabel('Date')
plt.ylabel("Sales")
plt.title("Monthly Customer Sales")
plt.show()


# In[15]:


#call the diferrence the sales colums to make the sales data stationery
monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna()
monthly_sales.head(10)


# In[32]:


plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'], monthly_sales['sales'])
plt.xlabel("date")
plt.ylabel('Sales')
plt.title("Monthly customer sales difference")
plt.show()


# In[16]:


#droppong of sales and date
supervised_data = monthly_sales.drop(['date', 'sales'], axis=1)


# In[17]:


# prepapering teh supervised data
for i in range(1,13):
    col_name = 'month_' + str(i)
    supervised_data[col_name] = supervised_data['sales_diff'].shift(i)
supervised_data = supervised_data.dropna().reset_index(drop=True)
supervised_data.head(10)


# In[18]:


#split data into train and test
train_data = supervised_data[:-12]
test_data = supervised_data[-12:]
print("Train data shape : ", train_data.shape)
print("Test data shape : ", test_data.shape)


# In[19]:


scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)


# In[20]:


x_train, y_train = train_data[:,1:], train_data[:,0:1]
x_test, y_test = test_data[:,1:], test_data[:,0:1]
y_train = y_train.ravel()
y_test = y_test.ravel()
print("X_train Shape : ", x_train.shape)
print("y_train Shape : ", y_train.shape)
print("x_test Shape : ", x_test.shape)
print("y_test Shape : ", y_test.shape)


# In[21]:


#make prediction data frame to merge the predicted sales prices of all trained algs
sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)


# In[22]:


act_sales = monthly_sales['sales'][-13:].to_list()
print(act_sales)


# In[23]:


#create linear regression model and predicted model
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_pre = lr_model.predict(x_test)


# In[27]:


lr_pre = lr_pre.reshape(-1,1)
lr_pre_test_set = np.concatenate([lr_pre, x_test], axis=1)
lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)


# In[29]:


result_list = []
for index in range(0, len(lr_pre_test_set)):
    result_list.append(lr_pre_test_set[index][0] + act_sales[index])
lr_pre_series = pd.Series(result_list, name="Linear Prediction")
predict_df = predict_df.merge(lr_pre_series, left_index = True, right_index = True)


# In[32]:


lr_mse = np.sqrt(mean_squared_error(predict_df['Linear Prediction'], monthly_sales['sales'][-12:]))
lr_mae = mean_absolute_error(predict_df['Linear Prediction'], monthly_sales['sales'][-12:])
lr_r2 = r2_score(predict_df['Linear Prediction'], monthly_sales['sales'][-12:])
print("Linear Regression MSE: ", lr_mse)
print("Linear Regression MAE: ", lr_mae)
print("Linear Regression R2: ", lr_r2)


# In[36]:


plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'], monthly_sales['sales'])
plt.plot(predict_df['date'], predict_df['Linear Prediction'])
plt.title("Customer sales Forecast using LR Model")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(["Actual Sales", "Predicted Sales"])
plt.show()


# In[ ]:




