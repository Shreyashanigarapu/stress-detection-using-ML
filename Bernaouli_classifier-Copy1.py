#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernouliNB
from sklearn.feature_extraction.text import CountVectorizer

# In[2]:


df=pd.read_csv('D:\spam.csv',encoding='latin-1')#loading data


# In[3]:


df.head(n=10)#visualize data


# In[4]:


df.head()


# In[5]:


#dimension of dataset
df.shape


# In[6]:


#5572 rows and 5 columns


# In[7]:


#preprocessing includes removing duplicates , empty cells.
#data imputation
#dropping columns with too many nan values
df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df.shape




# In[8]:


#check target values are binary or not
np.unique(df['class'])

# In[9]


np.unique(df['message'])

# In[10]
x=df['message'].values
y=df['class'].values
cv=CountVectorizer()
x=cv.fit_transform(x)
v=x.toarray()
print(v)


# In[11]:


first_col=df.pop('message')
df.insert(0,'message',first_col)
df


# In[12]:


train_x=x[:4179]
train_y=y[:4179]
test_x=x[4179:]
test_y=y[4179:]
bnb=BernoulliNB(binarize=0.0)
model=bnb.fit(train_x,train_y)
y_pred_train=bnb.predict(train_x)
y_pred_test=bnb.predict(test_x)


# In[13]:


print(bnb.score(train_x,train_y)*100)
print(bnb.score(test_x,test_y)*100)


# In[14]:


from sklearn.metrics import classification_report
print(classification_report(train_y,y_pred_train))


# In[15]:


from sklearn.metrics import classification_report
print(classification_report(test_y,y_pred_test))







