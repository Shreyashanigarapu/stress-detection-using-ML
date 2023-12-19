#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.naive_bayes import BernouliNB
from sklearn.feature_extraction.text import CountVectorizer


# In[3]:


import numpy as np#for linear algebra
import pandas as pd#for data analysis
import seaborn as sns#for data visualization
from matplotlib import pyplot as plt#for data visualization
from sklearn.naive_bayes import BernoulliNB#for Bernoulli NB implementation
from sklearn.feature_extraction.text import CountVectorizer#for spare matrix representation


# In[5]:


#loading data
df=pd.read_csv('D:\',encoding='latin-l')


# In[7]:


#loading data
df=pd.read_csv('D:\')


# In[13]:


df=pd.read_csv('D:\spam.csv',encoding='latin-1')#loading data


# In[14]:


df.head(n=10)#visualize data


# In[15]:


df.head()


# In[17]:


#dimension of dataset
df.shape


# In[18]:


#5572 rows and 5 columns


# In[22]:


#preprocessing includes removing duplicates , empty cells.
#data imputation
#dropping columns with too many nan values
df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df.shape


# In[23]:


df.head(n=4)


# In[24]:


#check target values are binary or not
np.unique(df['class'])


# In[25]:


#2 unique values so binary


# In[26]:


np.unique(df['message'])


# In[27]:


x=df["message"].values
y=df["class"].values
cv=CountVectorizer()
#transform values
x=cv.fit_transform(x)
v=x.toarray()
#print spare matrix
print(v)


# In[36]:


#data arrangement
#shift target cloumn to end 
first_col=df.pop('message')
df.insert(0,'message',first_col)
df


# In[37]:


df=pd.read_csv("D:\spam.csv",encoding="latin-1")


# In[38]:


df.head()


# In[42]:


df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df.shape


# In[43]:


np.unique(df['class'])


# In[44]:


x=df['message'].values
y=df['class'].values
cv=CountVectorizer()
x=cv.fit_transform(x)
v=x.toarray()
print(v)


# In[45]:


first_col=df.pop('message')
df.insert(0,'message',first_col)
df


# In[49]:


train_x=x[:4179]
train_y=y[:4179]
test_x=x[4179:]
test_y=y[4179:]
bnb=BernoulliNB(binarize=0.0)
model=bnb.fit(train_x,train_y)
y_pred_train=bnb.predict(train_x)
y_pred_test=bnb.predict(test_x)


# In[50]:


print(bnb.score(train_x,train_y)*100)
print(bnb.score(test_x,test_y)*100)


# In[51]:


from sklearn.metrics import classification_report
print(classification_report(train_y,y_pred_train))


# In[52]:


from sklearn.metrics import classification_report
print(classification_report(test_y,y_pred_test))


# In[ ]:




