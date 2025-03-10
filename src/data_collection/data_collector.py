import numpy as np
import pandas as pd
import os

def collect_data(dataframe, pos_x, pos_y, success, csv_name):
    df = dataframe
    df.loc[len(df)] = [len(df), pos_x, pos_y, success]

    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_name)
    df.to_csv(dir, index=False)
    
    return df

def find_csv(csv_name, dataframe):
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_name)

    if os.path.exists(dir):
        return pd.read_csv(dir, index_col=0)
    
    return dataframe