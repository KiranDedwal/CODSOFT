#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv(r"C:\Users\Dell\Downloads\Titanic-Dataset.csv")


# In[3]:


df


# In[7]:


len(df)


# In[8]:


df.shape


# In[13]:


df.index


# In[9]:


df.columns


# In[10]:


df.head()


# In[11]:


df.tail()


# In[12]:


df.info()


# In[13]:


df.dtypes   # to check the datatypes of columns


# In[14]:


df.isnull().sum()


# In[15]:


for i in df.columns:
    if(df[i].dtype=='object'):
        x=df[i].mode()[0]
        df[i]=df[i].fillna(x)
    else:
        x=df[i].mean()
        df[i]=df[i].fillna(x)


# In[9]:


df.describe()


# In[10]:


df.corr()


# In[12]:


df.duplicated().sum()


# In[16]:


# pclass 
print((df['Pclass'].value_counts()/891)*100)


# In[17]:


sns.countplot(df['Pclass'])


# In[17]:


# data analayis and visulization


# In[18]:


# countplot of survived vs not survived

sns.countplot(x='Survived',data=df)


# In[19]:


sns.countplot(x='Survived',data=df,hue='Sex')


# In[18]:


print(df['SibSp'].value_counts())
sns.countplot(df['SibSp'])


# In[23]:


# null value visulize
sns.heatmap(df.isna())


# In[24]:


# find the % of null value in age column
df['Age']


# In[26]:


df['Age'].isna().sum()


# #### Data Cleaning:

# In[27]:


df.isna().sum()


# In[28]:


# fill the age column by mean caluclating
df['Age'].mean()
df['Age'].fillna()


# In[29]:


df['Age'].fillna(df['Age'].mean(),inplace=True)


# In[31]:


df.isna().sum()


# In[21]:


sns.displot(df['Age'])


# In[32]:


sns.heatmap(df.isna())


# In[33]:


# not required cabin column so drop it
#axis=1 for column axis=0 for rows
df.drop('Cabin',axis=1,inplace=True)


# #### prepare the model

# In[37]:


# convert sex column  to numeric values
df['Sex']=df['Sex'].map({'female':0,'male':1}).inplace=True


# In[35]:


df


# In[45]:


# drop columns not required
#df.drop(['Name','Sex''Ticket','Embarked'],axis=1,inplace=True)


# In[40]:


df.head()


# In[42]:


df.shape


# In[46]:


# separate depenedent and independent varibales:


# In[47]:


# independent variable is y
# dependent varivales is x
#or ohter way to write
#x=df.drop(labels=['Survived','Name','Ticket','Cabin'],axis=1)
#y=df[[survived]]


# In[27]:


x=df[['PassengerId','Pclass','Age','SibSp','Fare']]
y=df['Survived']


# In[28]:


x


# In[29]:


y


# ### Data Modeling

# In[30]:


# building the model using logistic regression

#import train test split model
from sklearn.model_selection import train_test_split as tts


# In[31]:


# train test split

X_train,X_test,y_train,y_test=tts(x,y,test_size=0.30,random_state=40)


# In[32]:


print("X_TRAIN",X_train.shape)
print("X_TEST",X_test.shape)
print("Y_TRAIN",y_train.shape)
print("Y_TEST",y_test.shape)


# In[33]:


y_test


# In[34]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[35]:


X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[36]:


#Import logistic regression:
from sklearn.linear_model import LogisticRegression
LGR=LogisticRegression()


# In[37]:


LGR.fit(X_train,y_train)


# In[38]:


# Predict
y_pred=LGR.predict(X_test)


# In[39]:


y_pred


# In[40]:


# testing:


# In[41]:


from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,classification_report


# In[42]:


confusion_matrix(y_pred,y_test)


# In[43]:


#pd.DataFrame(confusion_matrix(y_test,y_pred),columns=['predicted no','predicted yes'],index=['actual no','actual yes'])


# In[44]:


accuracy_score(y_pred,y_test)


# In[45]:


precision_score(y_pred,y_test)


# In[46]:


recall_score(y_pred,y_test)


# In[73]:


f1_score(y_pred,y_test)


# In[75]:


print(classification_report(y_pred,y_test))


# In[ ]:




