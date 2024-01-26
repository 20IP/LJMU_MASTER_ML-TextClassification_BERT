import pandas as pd
import numpy as np
from logger_config import logger
class MedicalTextDataLoader:
    def __init__(self, args):
        self.data_path = args.data_path
        self.process_state = args.data_preprocess
        
    def load_data(self):
        logger.info(f"*** MedicalTextDataLoader: Data path: {self.data_path}")
        logger.info(f"*** MedicalTextDataLoader: Data preprocess: {self.process_state}")

        train_df = pd.read_csv(f'{self.data_path}/medical_tc_train.csv')
        test_df = pd.read_csv(f'{self.data_path}/medical_tc_test.csv')

        # Correctly shift labels by 1 (assuming they start from 1, not 0)
        train_df['condition_label'] = train_df['condition_label'] - 1
        test_df['condition_label'] = test_df['condition_label'] - 1
        num_labels = len(train_df['condition_label'].unique())

        if self.process_state:
            pass
        else:
            pass
        train_data = list(zip(train_df['medical_abstract'].tolist(), train_df['condition_label'].tolist()))
        test_data = list(zip(test_df['medical_abstract'].tolist(), test_df['condition_label'].tolist()))

        return train_data, test_data, num_labels