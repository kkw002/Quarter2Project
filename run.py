import sys
import json
import pandas as pd
sys.path.insert(0, 'src')
from etl import getData,getDataZip
from features import clean_features
from train_model import logistic_model

def main(targets):
    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)
        with open('config/holdoutdata_params.json') as fh:
            holddata_cfg = json.load(fh)
    if 'test' in targets:
        data = getData('data/testdata.pkl')
        df = clean_features(data)
        logistic_model(df)
    return 
if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)
