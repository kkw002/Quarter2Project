#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[14]:


complete_frame = pd.read_parquet('DSC180B.parquet')


# In[15]:


complete_frame
#consider the date where people buy things, whole dollar amounts


# In[17]:


complete_frame['memo']=complete_frame['memo'].apply(lambda x:x.lower())
complete_frame['memo']= complete_frame['memo'].apply(lambda x:re.sub(r'[^\w\s]','',x))
complete_frame['memo']=complete_frame['memo'].apply(lambda x:re.sub(r'[\d]','',x))


# In[34]:


split_set = complete_frame[['memo','new_category']]


# In[35]:


tfidf = TfidfVectorizer(stop_words='english') #1-3 character/words, 


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(split_set['memo'],split_set['new_category'],test_size=0.3, random_state=42)


# In[46]:


complete_frame[['memo','new_category']]


# In[49]:


lr=LogisticRegression() 


# In[54]:


tfidf.fit(X_train)


# In[56]:


x_train = tfidf.transform(X_train)
x_test = tfidf.transform(X_test)


# In[57]:


lr.fit(x_train,y_train)


# In[59]:


ypred = lr.predict(x_test)


# In[61]:


metrics.accuracy_score(y_test,ypred)


# In[ ]:




