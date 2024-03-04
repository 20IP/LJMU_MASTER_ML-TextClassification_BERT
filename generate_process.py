import os
import pandas as pd
import numpy as np
import re
from fuzzywuzzy import fuzz

from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from utils import *
import argparse
import warnings
import logging

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore') 

def process():
    parser = argparse.ArgumentParser(description="Data processing.")
    
    parser.add_argument("--data_dir",
                        type=str,
                        help="data path include train+test file",
                        default='../Medical-Abstracts-TC-Corpus')
    
    
    parser.add_argument("--file_name",
                        type=str,
                        required=True)
    
    parser.add_argument("--output_dir",
                        type=str,
                        default='../Medical-Abstracts-TC-Corpus',
                        required=True
                        )
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    file_name = args.file_name
    output_dir = f'{args.output_dir}/preprocessed-{file_name}'
    
    logging.info(">>>>>>>>> Data processing with: <<<<<<<<<< ")
    logging.info(f">>>>>>>>> Data directory: {data_dir}")
    logging.info(f">>>>>>>>> Data file name: {file_name}")
    logging.info(f">>>>>>>>> Data output: {output_dir}\n")
    
    df = pd.read_csv(f'{data_dir}/{file_name}')
    df.condition_label = df.condition_label.apply(lambda x: x-1)
    
    logging.info(f">>>>>>>>> Data size: {df.shape}")
    logging.info(f">>>>>>>>> Data duplicate: {df.medical_abstract.duplicated().sum()}\n")
    
    class_mapping = {"0": "Neoplasms",
                     "1": "Digestive system diseases",
                     "2": "Nervous system diseases",
                     "3": "Cardiovascular diseases",
                     "4": "General pathological conditions"}
    # Dissociate each data frame according to each class.
    df_class_0 = df[df.condition_label==0]
    df_class_1 = df[df.condition_label==1]
    df_class_2 = df[df.condition_label==2]
    df_class_3 = df[df.condition_label==3]
    df_class_4 = df[df.condition_label==4]

    df_class_0.reset_index(inplace=False, drop=True)
    df_class_1.reset_index(inplace=False, drop=True)
    df_class_2.reset_index(inplace=False, drop=True)
    df_class_3.reset_index(inplace=False, drop=True)
    df_class_4.reset_index(inplace=False, drop=True)
    
    # Calculate Jaccard score to measure the correlation between sequences.
    a_id_df_class_0_1, a_id_df_class_1, j_score0 = data_filter_parallel(df_class_0, df_class_1)
    a_id_df_class_0_2, a_id_df_class_2, j_score1 = data_filter_parallel(df_class_0, df_class_2)
    a_id_df_class_0_3, a_id_df_class_3, j_score2 = data_filter_parallel(df_class_0, df_class_3)
    a_id_df_class_0_4, a_id_df_class_4, j_score3 = data_filter_parallel(df_class_0, df_class_4)
    b_id_df_class_1_1, b_id_df_class_2, j_score4 = data_filter_parallel(df_class_1, df_class_2)
    b_id_df_class_1_2, b_id_df_class_3, j_score5 = data_filter_parallel(df_class_1, df_class_3)
    b_id_df_class_1_3, b_id_df_class_4, j_score6 = data_filter_parallel(df_class_1, df_class_4)
    c_id_df_class_2_1, c_id_df_class_3, j_score7 = data_filter_parallel(df_class_2, df_class_3)
    c_id_df_class_2_2, c_id_df_class_4, j_score8 = data_filter_parallel(df_class_2, df_class_4)
    d_id_df_class_3_1, d_id_df_class_4, j_score9 = data_filter_parallel(df_class_3, df_class_4)

    total_id0 = np.concatenate((a_id_df_class_0_1, a_id_df_class_0_2, a_id_df_class_0_3, a_id_df_class_0_4))
    total_id1 = np.concatenate((b_id_df_class_1_1, b_id_df_class_1_2, b_id_df_class_1_3))
    total_id2 = np.concatenate((c_id_df_class_2_1, c_id_df_class_2_2))
    total_id3 = d_id_df_class_3_1

    # Filter out unique indexes that are multi-label data.

    rep_df_class_0 = df_class_0.iloc[np.unique(total_id0)]
    rep_df_class_1 = df_class_1.iloc[np.unique(total_id1)]
    rep_df_class_2 = df_class_2.iloc[np.unique(total_id2)]
    rep_df_class_3 = df_class_3.iloc[np.unique(total_id3)]
    
    # Create a DataFrame containing values that are multi-labels.
    rep_df = pd.concat([rep_df_class_0, rep_df_class_1, rep_df_class_2, rep_df_class_3], ignore_index=True)
    rep_df = rep_df[['medical_abstract']]
    logging.info(f">>>>>>>>> Text selected by Jaccard: {rep_df.shape[0]}")
    
    
    rep_df.drop_duplicates(inplace=True)
    check_list = rep_df['medical_abstract'].tolist()
    f=0
    values_lbl = []
    
    # Convert data types and aggregate label data.
    for i in range(df.shape[0]):
        text = df['medical_abstract'].iloc[i]
        is_present = text in check_list
        if is_present:
            selected_df = df[df['medical_abstract']==text].condition_label.tolist()
        else:
            selected_df = [df.condition_label.iloc[i].tolist()]
        values_lbl.append(selected_df)

    # Create a new column named: category_condition_label.
    df['category_condition_label'] = values_lbl
    
    ### Convert condition_label to binary with multi-label
    
    logging.info(f">>>>>>>>> Converting multi-labelling:")
    
    mult_lbl = []
    for i in df.category_condition_label:
        mult_lbl.append(binary_cvt(i))
        
    df_bn = pd.DataFrame(mult_lbl, columns=['neoplasms', 'digestive', 'nervous', 'cardiovascular', 'general'])

    df = pd.concat([df_bn, df], axis=1)
    df.drop('category_condition_label', axis=1, inplace=True)
    
    df.drop_duplicates(subset='medical_abstract', inplace=True, keep='first')
    df.reset_index(drop=True, inplace=True)
    logging.info(f">>>>>>>>> After removed Duplicate values: {df.shape}")
    
    # Clear text by using stop words and top word cloud
    df = normalize_special_text(df, 'medical_abstract', 'normalize_medical_abstract')
    df = remove_stopwords(df, 'normalize_medical_abstract')

    top_0 = get_top_n_words(df[df.condition_label==0], cols='normalize_medical_abstract', n=50)
    top_1 = get_top_n_words(df[df.condition_label==1], cols='normalize_medical_abstract', n=50)
    top_2 = get_top_n_words(df[df.condition_label==2], cols='normalize_medical_abstract', n=50)
    top_3 = get_top_n_words(df[df.condition_label==3], cols='normalize_medical_abstract', n=50)
    top_4 = get_top_n_words(df[df.condition_label==4], cols='normalize_medical_abstract', n=50)
    
    com_words = find_common_words([top_0, top_1, top_2, top_3, top_4])
    com_words.update({"p": ''})
    
    # Utilize the lemma method for text processing.
    df['lemma_normalize_medical_abstract'] = df['medical_abstract'].apply(lambda x: stemming_lemma_reprocess(x, type_select='lemma'))
    top_0_lemma = get_top_n_words(df[df.condition_label==0], cols='lemma_normalize_medical_abstract', n=50)
    top_1_lemma = get_top_n_words(df[df.condition_label==1], cols='lemma_normalize_medical_abstract', n=50)
    top_2_lemma = get_top_n_words(df[df.condition_label==2], cols='lemma_normalize_medical_abstract', n=50)
    top_3_lemma = get_top_n_words(df[df.condition_label==3], cols='lemma_normalize_medical_abstract', n=50)
    top_4_lemma = get_top_n_words(df[df.condition_label==4], cols='lemma_normalize_medical_abstract', n=50)
    
    com_words_lemma = find_common_words([top_0_lemma, top_1_lemma, top_2_lemma, top_3_lemma, top_4_lemma])
    com_words_lemma.update({"p": ''})
    
    # Replace common words with empty at "normalize_medical_abstract" column.
    df.normalize_medical_abstract = df.normalize_medical_abstract.apply(lambda x: replace_words(x, com_words))

    # Replace common words with empty at "lemma_normalize_medical_abstract" column.
    df.lemma_normalize_medical_abstract = df.lemma_normalize_medical_abstract.apply(lambda x: replace_words(x, com_words_lemma))

    # Remove unnecessary columns.
    df.drop('condition_label', axis=1, inplace=True)    
    # Store the newly created data.
    df.rename(columns={"binary_condition_label": "condition_label"}, inplace=True)
    df.to_csv(output_dir, index=False, index_label=False)
    logging.info(f">>>>>>>>> COMPLETED <<<<<<<<")
    
process()