import os
import pandas as pd
import numpy as np

def process_dataset(args):
    data_path = args.data_path
    process_state = args.data_preprocess
    
    train_df = pd.read_csv(f'{data_path}/medical_tc_train.csv')
    test_df = pd.read_csv(f'{data_path}/medical_tc_test.csv')

    if process_state:
        pass
    else:
        pass
    train_df['condition_label'] = train_df['condition_label'] - 1
    test_df['condition_label'] = test_df['condition_label'] - 1
    print(train_df.head())
    print(test_df.head())
    
    train_data = list(zip(train_df['medical_abstract'].tolist(), train_df['condition_label'].tolist()))
    test_data = list(zip(test_df['medical_abstract'].tolist(), test_df['condition_label'].tolist()))
    
    return train_data, test_data