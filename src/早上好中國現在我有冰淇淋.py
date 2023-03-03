#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_extraction import text
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# In[2]:


complete_frame = pd.read_parquet('DSC180B.parquet')


# In[3]:


complete_frame
#consider the date where people buy things, whole dollar amounts


# In[4]:


complete_frame['memo']=complete_frame['memo'].apply(lambda x:x.lower())
complete_frame['memo']= complete_frame['memo'].apply(lambda x:re.sub(r'[^\w\s]','',x))
complete_frame['memo']=complete_frame['memo'].apply(lambda x:re.sub(r'[\d]','',x))


# In[27]:


#get states
states = [ 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
           'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
           'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
           'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
           'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
states = [x.lower() for x in states]


# In[38]:


stop_word_list =  text.ENGLISH_STOP_WORDS.union(["pos","card","purchase","debit"]).union(states)
ps= PorterStemmer()
complete_frame[['memo']]=complete_frame[['memo']].apply(lambda x:[ps.stem(w) for w in x]) 


# In[39]:


tfidf = TfidfVectorizer(stop_words=stop_word_list,ngram_range=(1,3),use_idf=True) #1-3 character/words, 


# In[40]:


split_set = complete_frame[['memo','new_category']]
X_train, X_test, y_train, y_test = train_test_split(split_set['memo'],split_set['new_category'],test_size=0.3, random_state=42)


# In[41]:


complete_frame[['memo','new_category']]


# In[42]:


lr=LogisticRegression()


# In[43]:


tfidf.fit(X_train)


# In[64]:


temp=pd.DataFrame(list(tfidf.vocabulary_.keys())[:300],columns=['word'])


# In[65]:


temp['idf_value']=temp['word'].apply(lambda x:tfidf.idf_[tfidf.vocabulary_[x]])


# In[66]:


print(temp.sort_values(by='idf_value',ascending=False).to_string())


# In[70]:


temp.loc[[212, 180, 0,90,2,53,97,58,156,14,88,231,144], :].sort_values(by='idf_value',ascending=True)


# In[14]:


x_train = tfidf.transform(X_train)
x_test = tfidf.transform(X_test)


# In[15]:


lr.fit(x_train,y_train)


# In[16]:


ypred = lr.predict(x_test)


# In[17]:


metrics.accuracy_score(y_test,ypred)


# # Adding Additional Features

# In[18]:


#use the transaction_date and convert into weekdays, ie through Monday through Sunday\
complete_frame['weekday']=complete_frame['transaction_date'].apply(lambda x:x.weekday())
complete_frame['month']=complete_frame['transaction_date'].apply(lambda x:x.month)


# In[19]:


split_set = complete_frame[['memo','new_category']]
tfidf = TfidfVectorizer(stop_words='english',ngram_range=(1,3)) #1-3 character/words, 
X_train, X_test, y_train, y_test = train_test_split(split_set['memo'],split_set['new_category'],test_size=0.3, random_state=42)


# In[20]:


tfidf.fit(X_train)


# In[21]:


x_train = tfidf.transform(X_train)
x_test = tfidf.transform(X_test)


# In[22]:


lr.fit(x_train,y_train)
ypred = lr.predict(x_test)
metrics.accuracy_score(y_test,ypred)


# In[23]:


complete_frame[complete_frame['new_category']=='Entertainment'].sort_values(by='amount')[-40:]


# In[ ]:





# In[ ]:




