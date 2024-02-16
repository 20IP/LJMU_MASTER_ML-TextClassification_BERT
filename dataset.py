import pandas as pd
import numpy as np
from logger_config import logger
import pandas as pd
import logging

class MedicalTextDataLoader:
    def __init__(self, args):
        """
        Initialize the MedicalTextDataLoader.

        Args:
        - data_path: Path to the data directory.
        - data_preprocess: Flag indicating whether data has been preprocessed.
        """
        self.data_path = args.data_path
        self.process_state = args.data_preprocess
        self.lemma = args.data_lemma
    
    def load_data(self, data_type='train'):
        """
        Load medical text data.

        Args:
        - data_type: Type of data to load ('train' or 'test').

        Returns:
        - data: A list of tuples containing preprocessed medical abstracts and condition labels.
        - num_labels: Number of unique condition labels.
        """
        logger.info(f"*** MedicalTextDataLoader: Data path: {self.data_path}")
        logger.info(f"*** MedicalTextDataLoader: Data preprocess: {'preprocessed-' if self.process_state else ''}medical_tc_{data_type}.csv")

        data_df = pd.read_csv(f'{self.data_path}/{"preprocessed-" if self.process_state else ""}medical_tc_{data_type}.csv')
        
        # Correctly shift labels by 1 (assuming they start from 1, not 0)
        data_df['condition_label'] = data_df['condition_label'] - 1

        num_labels = len(data_df['condition_label'].unique())
        
        if self.lemma is False:
            data = list(zip(data_df['lemma_normalize_medical_abstract'].tolist(), data_df['condition_label'].tolist()))
            logger.info("*** MedicalTextDataLoader: Lemmatization: True")
        else:
            data = list(zip(data_df['medical_abstract'].tolist(), data_df['condition_label'].tolist()))
            logger.info("*** MedicalTextDataLoader: Lemmatization: False")

        return data, num_labels


class MedicalTextDataEvalLoader(MedicalTextDataLoader):
    def __init__(self, data_path, data_preprocess):
        """
        Initialize the MedicalTextDataEvalLoader.

        Args:
        - data_path: Path to the data directory.
        - data_preprocess: Flag indicating whether data has been preprocessed.
        """
        super().__init__(data_path, data_preprocess)

    def load_data(self):
        """
        Load medical text evaluation data.

        Returns:
        - eval_data: A list of tuples containing preprocessed medical abstracts and condition labels for evaluation.
        """
        eval_data, _ = super().load_data(data_type='test')
        return eval_data
