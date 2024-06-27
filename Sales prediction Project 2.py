#!/usr/bin/env python
# coding: utf-8

# ### Sales Prediction Project:

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[4]:


df=pd.read_csv(r"C:\Users\Dell\Downloads\advertising (1).csv")
df.head()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


print(df.info())


# In[8]:


print(df.isnull().sum())


# In[9]:


df.describe()


# In[10]:


# duplicated
df.duplicated().sum()


# In[18]:


df.value_counts()


# In[12]:


sns.histplot(df,x="TV",kde=True)


# In[14]:


sns.histplot(df,x="Radio",kde=True)


# In[15]:


sns.histplot(df,x="Newspaper",kde=True)


# In[17]:


sns.heatmap(df.corr(),annot=True)


# In[20]:


sns.lineplot(x=df['TV'],y=df['Sales'])


# In[21]:


sns.lineplot(x=df['Radio'],y=df['Sales'])


# In[22]:


sns.lineplot(x=df['Newspaper'],y=df['Sales'])


# In[23]:


plt.figure(figsize=(10,15))
for i,cat in enumerate(df):
    ax=plt.subplot(7,2,i+1)
    sns.distplot(df[cat])
    plt.title(cat,fontsize=14,fontweight='bold')
plt.tight_layout()
plt.show()


# In[24]:


plt.figure(figsize=(10,20))
for i ,cat in enumerate(df):
    ax=plt.subplot(7,2,i+1)   # 2 subplot
    sns.boxplot(df[cat])
    plt.title(cat,fontsize=14,fontweight='bold')
plt.tight_layout()
plt.show()


# In[ ]:


# Feature selection and model selection


# In[25]:


x=df[['TV','Radio','Newspaper']]
y=df['Sales']


# In[26]:


x


# In[27]:


y


# In[28]:


from sklearn.model_selection import train_test_split as tts


# In[49]:


X_train, X_test, Y_train, Y_test = tts(x, y, test_size=0.2,random_state=42)


# In[50]:


print('shape of x_train:', X_train.shape)
print('shape of x_test:', X_test.shape)
print('shape of y_train:', Y_train.shape)
print('shape of y_test:', Y_test.shape)


# In[51]:


from sklearn.preprocessing import StandardScaler


# In[52]:


scaler = StandardScaler()

X_train_s= scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


# In[53]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()


# In[54]:


LR.fit(X_train,Y_train)


# In[55]:


y_pred=LR.predict(X_test)


# In[56]:


y_pred


# In[57]:


Y_test


# In[58]:


from sklearn.metrics import r2_score,mean_absolute_error as MAE,mean_squared_error as MSE


# In[59]:


score=r2_score(y_pred,Y_test)


# In[60]:


score


# In[61]:


print('Mean Absolute Error:',MAE(y_pred,Y_test))


# In[62]:


print('Mean Absolute Error:',MSE(y_pred,Y_test))


# In[63]:


sns.distplot(x=y_pred-Y_test,hist=False)


# I have used regressor models - Linear Regression model for predicting the sales.
# The Linear Regression has performed well with the accuracy of 89%

# In[ ]:




