import pandas as pd
import numpy as np
from logger_config import logger
import pandas as pd
import ast
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
        self.based_process = args.based_process
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
        logger.info(f"*** MedicalTextDataLoader: Data preprocess: preprocessed-medical_tc_{data_type}.csv")

        data_df = pd.read_csv(f'{self.data_path}/preprocessed-medical_tc_{data_type}.csv', engine='python')
        data_df[['neoplasms', 'digestive', 'nervous', 'cardiovascular', 'general']] = data_df[['neoplasms', 'digestive', 'nervous', 'cardiovascular', 'general']].astype(int)
        num_labels = 5

        if self.based_process is False and self.lemma is False:
            data = [data_df['medical_abstract'].tolist(), data_df[['neoplasms', 'digestive', 'nervous', 'cardiovascular', 'general']].values]
            logger.info("*** MedicalTextDataLoader: Loaded and process with 'medical_abstract' column")
            
        elif self.based_process is True and self.lemma is False:
            data = [data_df['normalize_medical_abstract'].tolist(), data_df[['neoplasms', 'digestive', 'nervous', 'cardiovascular', 'general']].values]
            logger.info("*** MedicalTextDataLoader: Loaded and process with 'normalize_medical_abstract' column")
            
        elif self.based_process is False and self.lemma is True:
            data = [data_df['lemma_normalize_medical_abstract'].tolist(), data_df[['neoplasms', 'digestive', 'nervous', 'cardiovascular', 'general']].values]
            logger.info("*** MedicalTextDataLoader: Loaded and process with 'lemma_normalize_medical_abstract' column")
        
        else:
            raise ValueError('The values of "based_process" and "lemma" must not be True simultaneously.')

        return data, num_labels


class MedicalTextDataEvalLoader(MedicalTextDataLoader):
    def __init__(self, data_preprocess):
        """
        Initialize the MedicalTextDataEvalLoader.

        Args:
        - data_path: Path to the data directory.
        - data_preprocess: Flag indicating whether data has been preprocessed.
        """
        super().__init__(data_preprocess)

    def load_data(self):
        """
        Load medical text evaluation data.

        Returns:
        - eval_data: A list of tuples containing preprocessed medical abstracts and condition labels for evaluation.
        """
        eval_data, num_labels = super().load_data(data_type='test')
        return eval_data, num_labels
