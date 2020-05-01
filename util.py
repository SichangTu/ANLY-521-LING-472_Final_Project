from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import string

def preprocess_text(text):
    """Preprocess one sentence: tokenizes, lowercases, applies the Porter stemmer,
     removes punctuation tokens and stopwords.
     Returns a string of tokens joined by whitespace."""
    text_split = text.split("\t")
    label = text_split[0]
    if label == 'ham':
        label = 0
    else:
        label = 1
    msg = text_split[1]
    tokens = word_tokenize(msg.lower())
    stemmed_tokens = [PorterStemmer().stem(w) for w in tokens]
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in stemmed_tokens if w not in stop_words]
    no_punc = [w for w in filtered if w not in string.punctuation]
    msg = " ".join(no_punc)
    return label, msg


def shuffle_dataset(data, labels):
    """
    Shuffles a list of datapoints and their labels in unison
    :param data: iterable, each item a datapoint
    :param id_strs: iterable, each item an id
    :return: tuple (shuffled_data, shuffled_id_strs)
    """
    new_order = np.random.permutation(len(labels))
    # flexible to work with lists (i.e. of strings) or np
    if isinstance(labels, np.ndarray):
        shuffled_labels = labels[new_order]
    else:
        shuffled_labels = [labels[i] for i in new_order]
    if isinstance(data, np.ndarray):
        shuffled_data = data[new_order]
    else:
        shuffled_data = [data[i] for i in new_order]
    return shuffled_data, shuffled_labels


def split_data(X, labels, test_percent=0.25, shuffle=True):
    """
    Splits dataset for supervised learning and evaluation
    :param X: iterable of features
    :param labels: iterable of file id's corresponding the features in X
    :param test_percent: percent data to
    :param shuffle:
    :return: two tuples, (X_train, file_ids_train), (X_test, file_ids_test)
    """
    if shuffle:
        X, labels = shuffle_dataset(X, labels)
    data_size = len(X)
    num_test = int(test_percent * data_size)

    train = (X[:-num_test], labels[:-num_test])
    test = (X[-num_test:], labels[-num_test:])
    return train, test