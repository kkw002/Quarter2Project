import pandas as pd
def clean_features(df):
  df['memo']=df['memo'].apply(lambda x:x.lower())
  df['memo']= df['memo'].apply(lambda x:re.sub(r'[^\w\s]','',x))
  df['memo']=df['memo'].apply(lambda x:re.sub(r'[\d]','',x))
  return df
