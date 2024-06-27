#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[23]:


df=pd.read_csv(r"C:\Users\Dell\Downloads\archive (1)\IRIS.csv")
df.head()


# In[24]:


df.shape


# In[25]:


df.info


# In[26]:


df.describe()


# In[27]:


df.isnull().sum()


# In[32]:


df.duplicated().sum()


# In[33]:


df.drop_duplicates()


# In[4]:


df[df['sepal_width']>4]


# In[5]:


df[df['petal_width']>1]


# In[6]:


df[df['petal_width']>2]


# In[12]:


sns.scatterplot(x='sepal_length',y='petal_length',data=df,hue='species')
plt.show()


# Above plot seen like linear regression ...as sepal length increase petal length is also increasing...from plt we can see setosa has small petal length and sepal length and versicolor has in medial of both nd virginica has largest lenths.

# In[3]:


plot=plt.figure(figsize=(5,5))     
sns.countplot(x='species',data=df)


# In[4]:


plt.figure(figsize=(4,4))
sns.heatmap(df.corr(),annot=True)


# In[13]:


df.head()


# In[34]:


# feature selection nd training the model:


# In[37]:


x=df.drop('species',axis=1)
x


# In[39]:


y=df['species']
y


# In[42]:


#building the model using logistic regression

#import train test split model

from sklearn.model_selection import train_test_split as tts


# In[79]:


# train test split

X_train,X_test,y_train,y_test=tts(x,y,test_size=0.20,random_state=5)


# In[80]:


print("X_TRAIN",X_train.shape)
print("X_TEST",X_test.shape)
print("Y_TRAIN",y_train.shape)
print("Y_TEST",y_test.shape)


# In[81]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[82]:


X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[83]:


#Import logistic regression:
from sklearn.linear_model import LogisticRegression
LGR=LogisticRegression()


# In[84]:


LGR.fit(X_train,y_train)


# In[85]:


y_pred=LGR.predict(X_test)


# In[86]:


y_pred


# In[87]:


y_test


# In[88]:


from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,classification_report


# In[89]:


accuracy_score(y_pred,y_test)


# In[67]:


print(classification_report(y_pred,y_test))


# In[ ]:




