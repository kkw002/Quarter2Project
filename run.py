import sys
import json
import pandas as pd
sys.path.insert(0, 'src')
from etl import getData
from features import clean_features
from train_model import logistic_reg

def main(targets):
    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
    if 'test' in targets:
        data = getData('data/testdata.parquet')
        df = clean_features(data)
        logistic_reg(df)
    return 
if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
