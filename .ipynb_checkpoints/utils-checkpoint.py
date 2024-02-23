import os
import numpy as np
from tqdm import tqdm
import re

import plotly.graph_objs as go
from plotly.offline import iplot

import torch
from transformers import BertTokenizer

from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from multiprocessing import Pool
from collections import Counter

# Download stop words from the NLTK library for the English language.
nltk.download('stopwords')


def jaccard_similarity_multi_sentences(input1, input2):
    '''
    Calculate Jaccard similarity between two sentences.

    Parameters:
    - input1 (str): First sentence.
    - input2 (str): Second sentence.

    Returns:
    float: Jaccard similarity score between the two sentences.
    '''
    sentence1 = set(input1.replace('.', ' ').lower().split())
    sentence2 = set(input2.replace('.', ' ').lower().split())
    cm = sentence1.intersection(sentence2)
    score = float(len(cm)) / (len(sentence1) + len(sentence2) - len(cm))
    
    return score

def parallel_process_data_filter(args):
    '''
    Parallel processing function to calculate Jaccard similarity scores.

    Parameters:
    - args (tuple): Tuple containing (df1, df2, col, i).

    Returns:
    tuple: Tuple containing numpy arrays of indices and scores.
    '''
    df1, df2, col, i = args
    idx_1, idx_2, scores = [], [], []
    for j in range(len(df2)):
        j_score = jaccard_similarity_multi_sentences(df1[col].iloc[i], df2[col].iloc[j])
        if j_score > 0.8:
            idx_1.append(i)
            idx_2.append(j)
            scores.append(j_score)
    return np.array(idx_1), np.array(idx_2), np.array(scores)

def data_filter_parallel(df1, df2, col='medical_abstract', processes=8):
    '''
    Filter data in parallel based on Jaccard similarity scores.

    Parameters:
    - df1 (DataFrame): First DataFrame.
    - df2 (DataFrame): Second DataFrame.
    - col (str): Column name for text data.
    - processes (int): Number of parallel processes.

    Returns:
    tuple: Tuple containing numpy arrays of indices and scores.
    '''
    pool = Pool(processes)
    func_args = [(df1, df2, col, i) for i in range(len(df1))]
    results = []

    with tqdm(total=len(func_args), desc="Processing", position=0) as pbar:
        for result in pool.imap_unordered(parallel_process_data_filter, func_args):
            results.append(result)
            pbar.update()

    pool.close()
    pool.join()

    idx_1, idx_2, scores = [], [], []
    for result in results:
        idx_1.extend(result[0])
        idx_2.extend(result[1])
        scores.extend(result[2])

    return np.array(idx_1), np.array(idx_2), np.array(scores)

def check_replace_values(score):
    '''
    Check if all values in the array are the same.

    Parameters:
    - score (numpy array): Array of values.

    Prints:
    - Information about whether all values are the same or not.
    '''
    equal_or_not = np.count_nonzero((score == 1)) == len(score)
    if not equal_or_not:
        print('Is the same :', equal_or_not, '\t', 'Other values 1: ', score[score!=1])
    else:
        print('Is all the same :', equal_or_not, '\t', 'Total values: ', len(score))

def visualize_multilabel(label,
                         data,
                         title,
                         x_title = 'Description',
                         color='darkslategray'):
    
    fig = go.Figure(data=[go.Bar(x=label,
                                 y=data,
                                 marker_color=color,
                                 text=data,
                                 textposition='auto')])
    fig.update_layout(
        # xaxis=dict(tickangle=-30),
        xaxis_title=x_title,
        yaxis_title='Values',
        coloraxis_colorbar=dict(title='Values'),
        bargap=0.1,
        title=title,
        plot_bgcolor='lightgray'
    )
    fig.show()


def normalize_special_text(df, column_name, new_col, characters_to_remove=None, char_mapping=None):
    """
    Normalize the specified text column in the DataFrame by removing specified characters and applying mapping.

    Parameters:
    - df: DataFrame
        The DataFrame containing the text column to normalize.
    - column_name: str
        The name of the text column to normalize.
    - characters_to_remove: list, optional
        List of characters to remove from the text.
    - char_mapping: dict, optional
        Dictionary mapping characters to their normalized replacements.

    Returns:
    - DataFrame
        The DataFrame with the normalized text column.
    """
        
    characters_to_remove = characters_to_remove or ['$', '!', '?', '(', ')', '[', ']', '"', "'", '=', ':', '`', ',', '.']
    char_mapping = char_mapping or {'$': '', '!': '', '?': '', '(': '', ')': '', '[': '',
                                    ']': '', '"': '', "'": '', '=': ' ', ':': '', '`': '', ',': '', '.':''}

    pattern = re.compile('|'.join(re.escape(char) for char in characters_to_remove))
    df[new_col] = df[column_name].apply(lambda x: pattern.sub(lambda m: char_mapping.get(m.group(0), ''), x))
    df[new_col] = df[new_col].apply(lambda x: re.sub(r"\s+", " ", x).strip())   # Remove extra whitespaces
    return df

def remove_stopwords(df, column_name):
    """
    Remove stop words from a text column in a DataFrame.

    Parameters:
    - df: DataFrame
        The DataFrame containing the text column.
    - column_name: str
        The name of the text column to remove stop words from.

    Returns:
    - DataFrame
        The DataFrame with stop words removed from the specified text column.
    """

    # Retrieve the set of English stop words from the NLTK library
    words_to_keep = {'y', 'x', 'll'}
    
    stop_words = set(stopwords.words('english'))
    stop_words -= words_to_keep

    # Apply the stop word removal process to the specified text column
    df[column_name] = df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    return df


def show_wordCount(df, col_name, idx, lbl):
    """
    Generate a histogram and print statistics for word count distribution for a specific condition.

    Parameters:
    - df: DataFrame
        The DataFrame containing the text data and condition labels.
    - idx: int
        The condition label for which the word count distribution is to be visualized.
    - lbl: str
        The label corresponding to the condition for better identification in the plot title.

    Returns:
    - None
    """
    df[col_name][df['condition_label'] == idx].iplot(
        kind='hist',
        bins=150,
        xTitle='Text length',
        linecolor='black',
        color='red',
        yTitle='Vocabulary Frequency',
        title=f'{lbl} words count distribution'
    )

    # Print statistics
    print(f'Max word count for {lbl}: {max(df[col_name][df.condition_label==idx])}')
    print(f'Min word count for {lbl}: {min(df[col_name][df.condition_label==idx])}')


def get_quartile_range(df, col, id_list):
    """
    Generate box plots to show the quartile range of word count distribution for multiple conditions.

    Parameters:
    - df: DataFrame
        The DataFrame containing the text data and condition labels.
    - id_list: list
        List of condition labels for which box plots will be generated.

    Returns:
    - None
    """
    color_list = ['blueviolet', 'olive', 'lightgreen', 'hotpink', 'red']
    
    # Create box plots
    trace = [go.Box(y=df[col][df['condition_label']==idx],
                    name=f'{class_mapping[str(idx)]}',
                    marker=dict(color=color_list[idx])
                    ) for idx in id_list]

    layout = go.Layout(title="Length of the text", xaxis=dict(title='Condition Label'), yaxis=dict(title='Word Count'))
    fig = go.Figure(data=trace, layout=layout)
    iplot(fig, filename="Quartile of word count distribution")
    
def visualize_wordcloud(text_clean, ax, title, max_words):
    """
    Generate and visualize a word cloud from cleaned text data.

    Parameters:
    - text_clean: list
        List of cleaned text data.
    - ax: matplotlib.axes._subplots.AxesSubplot
        Matplotlib axis on which the word cloud will be visualized.
    - title: str
        Title for the word cloud visualization.

    Returns:
    - None
    """
    # Create a WordCloud object with specified settings
    wordcloud = WordCloud(
        background_color='white',
        width=480,
        height=480,
        max_words=max_words
    ).generate(" ".join(text_clean))

    # Display the word cloud on the provided axis
    ax.imshow(wordcloud)
    
    # Turn off axis labels
    ax.axis('off')
    
    # Set the title for the word cloud visualization
    ax.set_title(title, fontsize=20)

def get_clean_text(data):
    """
    Clean and preprocess text data by joining and splitting.

    Parameters:
    - data: list
        List of text data to be cleaned.

    Returns:
    - list
        Cleaned and preprocessed text data.
    """
    # Join the text data into a single string and then split into a list of words
    data = ' '.join(data).strip().split(' ')
    return data

def get_top_n_words(df, cols, n=None):
    """
    Get the top N most frequent words from a specific column in a DataFrame.

    Parameters:
    - df: DataFrame
        The DataFrame containing the preprocessed text column.
    - column_name: str
        The name of the column containing preprocessed text (without stop words).
    - n: int, optional
        Number of top words to retrieve. If None, return all words.

    Returns:
    - list
        A list of tuples containing the top N words and their frequencies.
    """
    # Extract the preprocessed text column from the DataFrame
    corpus = df[cols].tolist()

    vectorizer = CountVectorizer().fit(corpus)
    bag_of_words = vectorizer.transform(corpus)

    sum_words = bag_of_words.sum(axis=0)

    # Create a list of tuples containing word and frequency
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    return words_freq[:n]

def find_common_words(words_freq_lists, relative_n = 3):
    """
    Find common words that appear in at least 3 out of 5 words_freq lists.

    Parameters:
    - words_freq_lists: list
        List of words_freq lists, each containing tuples of (word, frequency).

    Returns:
    - list
        List of common words meeting the specified criteria.
    """
    sets_of_words = [list(zip(*words_freq))[0] for words_freq in words_freq_lists]
    combined_list = [value for tpl in sets_of_words for value in tpl]

    word_counts = Counter(combined_list)
    rare_words = {key: '' for key, value in word_counts.items() if value >= 3}
    return rare_words

def binary_cvt(labels, max_val=4):
    """
    Converts a list of labels into a binary representation using one-hot encoding.

    Parameters:
    - labels (list): List of integer labels.
    - max_val (int): Maximum possible value of the labels. Default is 4.

    Returns:
    list: Binary array representing the presence or absence of each label.
    """
    # Create an array of zeros with a length of (max_val + 1)
    zeros_arr = [0] * (max_val + 1)

    # Iterate through each label in the input list
    for label in labels:
        # Set the corresponding index in the array to 1, indicating the presence of the label
        zeros_arr[label] = 1

    # Return the resulting binary array
    return zeros_arr

def replace_words(text, replacements):
    pattern = re.compile(r'\b(?:%s)\b' % '|'.join(map(re.escape, replacements.keys())), re.IGNORECASE)
    return pattern.sub(lambda x: replacements[x.group().lower()], text)

def visualization_top_word_count(x_data, y_data, text_data, color=None, title=None, top_n=None):
    """
    Generate a bar chart using Plotly.

    Parameters:
    - x_data: list
        X-axis data.
    - y_data: list
        Y-axis data.
    - text_data: list
        Text data for hover text.
    - color: str, optional
        Bar color.
    - title: str, optional
        Chart title.
    - top_n: int, optional
        Number of top words to display.

    Returns:
    - go.Figure
        Plotly Figure object.
    """
    if top_n:
        x_data = x_data[:top_n]
        y_data = y_data[:top_n]
        text_data = text_data[:top_n]

    fig = go.Figure([go.Bar(x=x_data, y=y_data, text=text_data, marker_color=color)])
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', title_text=title)
    # fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    return fig

def stemming_lemma_reprocess(text, type_select='stemming'):
    """
    Preprocesses text by applying stemming or lemmatization.

    Parameters:
    - text (str): Input text to be preprocessed.
    - type_select (str, optional): Type of preprocessing to apply. 
      Options: 'stemming' for stemming, 'lemma' for lemmatization.
      Default is 'stemming'.

    Returns:
    - str: Processed text after applying stemming or lemmatization.
    """
    # Tokenize the input text
    words = word_tokenize(text)
    
    # Apply stemming or lemmatization based on type_select
    if type_select == 'stemming':
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    elif type_select == 'lemma':
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    else:
        # Raise an error for invalid type_select values
        raise ValueError('type_select must be either "stemming" or "lemma"')
        
        
def tokenize_and_format(data, direction_model, device='cpu'):
    """
    Tokenizes and formats the input data for training or evaluation.

    Args:
        data (list): List of tuples containing text data and corresponding labels.
        direction_model (str): Path or identifier of the pretrained model.
        device (str, optional): Device to which tensors are moved ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        dict: A dictionary containing the tokenized and formatted inputs for the model.
    """
    # Load the tokenizer for the specified pretrained model
    tokenizer = BertTokenizer.from_pretrained(direction_model)

    # Tokenize the input text data
    inputs = tokenizer(
        [item for item in data],
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512,
        return_token_type_ids=True
    )

    # Return a dictionary containing the tokenized and formatted inputs
    return tokenizer, ({
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'token_type_ids': inputs['token_type_ids']
    })

def calculate_metrics(true_labels, predicted_labels, threshold=0.5, report_methods=['micro', 'macro']):
    # Convert probability scores to binary predictions based on the threshold
    binary_predictions = (predicted_labels > threshold).astype(int)

    metrics = {}

    if 'micro' in report_methods:
        # Micro-Averaging: Calculate metrics globally across all classes
        tp = np.sum((true_labels == 1) & (binary_predictions == 1))
        fp = np.sum((true_labels == 0) & (binary_predictions == 1))
        fn = np.sum((true_labels == 1) & (binary_predictions == 0))
        tn = np.sum((true_labels == 0) & (binary_predictions == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0


    if 'macro' in report_methods:
        # Macro-Averaging: Calculate metrics independently for each class and then average
        num_classes = true_labels.shape[1]
        precision = recall = f1_score = accuracy = 0

        for class_idx in range(num_classes):
            tp = np.sum((true_labels[:, class_idx] == 1) & (binary_predictions[:, class_idx] == 1))
            fp = np.sum((true_labels[:, class_idx] == 0) & (binary_predictions[:, class_idx] == 1))
            fn = np.sum((true_labels[:, class_idx] == 1) & (binary_predictions[:, class_idx] == 0))
            tn = np.sum((true_labels[:, class_idx] == 0) & (binary_predictions[:, class_idx] == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            precision += precision
            recall += recall
            f1_score += f1_score
            accuracy += accuracy

        precision /= num_classes
        recall /= num_classes
        f1_score /= num_classes
        accuracy /= num_classes

    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1_score
    metrics['accuracy'] = accuracy

    return metrics