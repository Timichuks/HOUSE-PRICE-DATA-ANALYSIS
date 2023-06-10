#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score


# In[2]:


# import file
df=pd.read_csv('houseprice_data.csv')
# details of rows and column
df.head()


# In[3]:


df.info()
df.isnull().sum()


# In[4]:


# No missing valuesin the dataset


# In[5]:


# Data Exploration


# In[6]:


df.describe()
# statistical summary


# In[7]:


# showing correlation between the features
df.corr()


# In[8]:


plt.figure(figsize=(16,11))
sns.heatmap(df.corr(),annot=True)


# In[9]:


df.columns


# In[10]:


sns.scatterplot(x=df.bedrooms,y=df.price,color="grey")


# In[11]:


sns.set(style='dark')
sns.countplot(x=df.bedrooms,palette='deep')


# In[12]:


#dataset is populated by 3 bedroom


# In[13]:


# trend of house prices per year
sns.lineplot(x=df.yr_built,y=df.price,color='orange')


# In[14]:


# 1940 shows that house prices were low 


# In[15]:


y =df['price']


# In[16]:


fig1, ax= plt.subplots(2,3,figsize=(16,10))
ax[0,0].scatter(df.iloc[:,2],y,color='green')
ax[0,0].set_xlabel('bathroom')
ax[0,0].set_ylabel('Price')
ax[0,1].scatter(df.iloc[:,3],y)
ax[0,1].set_xlabel('condition')
ax[0,1].set_ylabel('Price')
ax[0,2].scatter(df.iloc[:,9],y)
ax[0,2].set_xlabel('grade')
ax[0,2].set_ylabel('Price')
ax[1,0].scatter(df.iloc[:,10],y)
ax[1,0].set_xlabel('sqft_basement')
ax[1,0].set_ylabel('Price')
ax[1,1].scatter(df.iloc[:,11],y)
ax[1,1].set_xlabel('sqft_living15')
ax[1,1].set_ylabel('Price')


# In[17]:


#comparing 2 variables
sns.scatterplot(x=df.sqft_living,y=df.price,color='brown')


# In[18]:


# As he sqftliving increases the price of the houses increases. this shows that increasing this can help to make a more informed decision 


# In[19]:


sns.scatterplot(x=df.sqft_above,y=df.price,color='black')


# In[ ]:





# In[20]:


x=df


# In[21]:


y=df.iloc[:,0].values


# In[22]:


y[0:5]


# In[23]:


x.shape


# In[24]:


y.shape


# In[25]:


x=df.iloc[:,3].values
y=df.iloc[:,0].values


# In[26]:


# splitting of train and test set


# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
print('x-train.shape:',x_train.shape)
print('x-test.shape:',x_test.shape)
print('y-train.shape:',y_train.shape)
print('y-test.shape:',y_test.shape)


# In[28]:


x_train=x_train.reshape(-1,1)
x_test=x_test.reshape(-1,1)


# In[29]:


regr = LinearRegression()
regr.fit(x_train,y_train)


# In[30]:


pred=regr.predict(x_test)
# mean_squared_error
MSE=mse(y_test,pred)
MSE


# In[31]:


# coefficients
print('coefficients:',regr.coef_)


# In[32]:


# intercept
print('intercept:',regr.intercept_)


# In[33]:


#cofficient of determination(performance evaluation)
print('cofficient of determination:%2f'%r2_score(y_test,regr.predict(x_test)))


# In[34]:


# visualize testing data 


# In[35]:


fig1,ax1=plt.subplots()
ax1.scatter(x_test,y_test,color='blue')
ax1.plot(x_test,regr.predict(x_test),color='red')
ax1.set_xlabel('sqft_living')
ax1.set_ylabel('price')
fig1.tight_layout()


# In[36]:


fig2,ax2=plt.subplots()
ax2.scatter(x_train,y_train,color='blue')
ax2.plot(x_train,regr.predict(x_train),color='red')
ax2.set_xlabel('sqft_living')
ax2.set_ylabel('price')
fig2.tight_layout()


# In[37]:


x=df[['sqft_living','grade','bathrooms']]
x.shape


# In[39]:


# more features
x=df[['sqft_living','sqft_above','grade','sqft_living15']].values
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
regr=LinearRegression()
regr.fit(x_train,y_train)


# In[41]:


print('coefficients:',regr.coef_)
print('intercept:',regr.intercept_)
print('mean squared error:%8f'% mse(y_test,regr.predict(x_test)))
print('coefficient of determine:%.2f'% r2_score(y_test,regr.predict(x_test)))
    


# In[42]:


x= df.iloc[:,1:].values
y=df.iloc[:,0].values


# In[44]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=2)
regr=LinearRegression()
regr.fit(x_train,y_train)
print('coefficients:',regr.coef_)
print('intercept:',regr.intercept_)
print('mean squared error:%8f'%mse(y_test,regr.predict(x_test)))
print('coefficient of determination:%.2f'% r2_score(y_test,regr.predict(x_test)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




