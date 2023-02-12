import pandas as pd

def getData(fp):
  data = pd.read_parquet(fp)
  return data
