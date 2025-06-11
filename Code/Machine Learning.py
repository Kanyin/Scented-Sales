#!/usr/bin/env python
# coding: utf-8

# ### Ebay Pefume Sales Prediction using LightGBM

# Here we predict sales based on brand, pricing, gender, and other factors. We will be using well known LightGBM, a machine learning tool. It will help to estimate how likely a brand is to sell their product even before it's listed or stocked.




import pandas as pd
import numpy as np
import os
import re
import plotly
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#import xgboost as xgb
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor as lgb
from lightgbm import early_stopping, log_evaluation





os.chdir("/Users/kanyin/Documents/Kanyin/Data Science/Data Projects/Perfume E-Commerce")
perf=pd.read_csv("Data/perf.csv")
perf=perf.drop('brand', axis=1)
perf.rename(columns={'brand.1':'brand'},inplace=True)
perf.head()





# Removing variables that may stunt prediction
perf=perf.drop(['title','lastUpdated','available','Available', 'sold'], axis=1)
perf





#check for NA values out of the 2000 entries
print(perf.isnull().sum())





perf.head()





perf['size'] = perf['size'].fillna(perf['size'].median())
perf['type'] = perf['type'].fillna(perf['type'].mode()[0])
#remove those in the target column
perf = perf[perf['Sold'].notna()]


# In[ ]:


import seaborn as sns

plt.figure





catgry= ['brand','type', 'Sex', 'itemLocation']
for col in catgry:
    perf[col] = perf[col].astype('category')





# Split the data so that we use 20% to test, 80% to train
A= perf.drop(['Sold'], axis=1)
B= perf['Sold']
A_train, A_test, B_train, B_test = train_test_split(A,B, test_size=0.2, random_state=42)





# Define parameters
params= { 'objective':'regression',
          'metric': 'rmse',
          'boosting_type': 'gbdt',
          'learning_rate': 0.05,
          'num_leaves': 15,
          'min_data_in_leaf' : 20,
          'feature_fraction': 0.8,
          'bagging_fraction': 0.8,
          'bagging_freq': 1,
          'random_state':42
        }

#Fit the model
model = LGBMRegressor(**params)
model.fit(
    A_train, B_train,
    eval_set=[(A_test, B_test)],
    callbacks=[early_stopping(stopping_rounds=20), log_evaluation(period=10)],
    #verbose=False
)





#Evaluate the model
B_pred = model.predict(A_test)

# Compute RMSE
mse = mean_squared_error(B_test, B_pred)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MSE: {mse:.2f}")





import lightgbm as gbm
import matplotlib.pyplot as plt
gbm.plot_importance(model, max_num_features=10)
plt.show()





print(perf['Sold'].describe())





perf





import plotly.express as px
data=perf.groupby(['brand','Sex'])['Sold'].sum().reset_index()
data
fig=px.bar(data, x='brand', y='Sold', color='Sex', title='Brand Sales by Sex')
fig.show()





import matplotlib.pyplot as plt
perf['log_Sold']=np.log1p(perf['Sold']) #log transform the sales as original sales data is skewed to the right
perf['log_Sold'].hist(bins=50)
plt.title("Distribution of number_sold")





model.fit(A_train, perf.loc[A_train.index, 'log_Sold'])
pred_log=model.predict(A_test)
pred = np.expm1(pred_log)
pred





A = perf.drop(columns=['Sold', 'log_Sold'])
B = perf['log_Sold']

A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.2, random_state=42)
model.fit(
    A_train, B_train,
    eval_set=[(A_test, B_test)],
    callbacks=[early_stopping(stopping_rounds=20), log_evaluation(period=10)],
    #verbose=False
)





#Back transform
B_pred_log = model.predict(A_test)
B_pred = np.expm1(B_pred_log)  # Inverse of log1p to get original scale predictions

# Compute RMSE
mse = mean_squared_error(B_test, B_pred)
rmse = np.sqrt(mse)

print(f"Test RMSE: {rmse:.2f}")
print(f"Test MSE: {mse:.2f}")





mse= mean_squared_error(perf.loc[B_test.index, 'Sold'],B_pred)
print(f"Validation RMSE: {rmse:.2f}")





from sklearn.cluster import KMeans

features = perf[['Sold', 'Brand']]  # Add more if desired
kmeans = KMeans(n_clusters=3)
perf['cluster'] = kmeans.fit_predict(features)


# It doesn't seem that our machine learning model really captures the nature of the sales. Let's switch to clustering given that
# most of the bulk sales are well over 1500 in an instance. It's likely that many buyers are most likely distributers.





