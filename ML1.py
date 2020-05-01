#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
#from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from util import preprocess_text, shuffle_dataset, split_data
import string
import argparse

np.random.seed(93)

import nltk
nltk.download('punkt')
nltk.download('stopwords')

#data_file = 'SMSSpamCollection.txt'

def main(datafile):
    with open(datafile, encoding='utf-8') as f:
        texts = f.read().splitlines()

    labels = []
    corpus = []
    for text in texts:
        label, msg = preprocess_text(text)
        labels.append(label)
        corpus.append(msg)

    train, test = split_data(corpus, labels, 0.2)
    y_train = np.asarray(train[1]).astype('int32').reshape((-1, 1))
    y_test = np.asarray(test[1]).astype('int32').reshape((-1, 1))


# feature extractions

def count_vec(train, test):
    count_vector = CountVectorizer()
    train_vec = count_vector.fit_transform(train)
    test_vec = count_vector.transform(test)
    return train_vec, test_vec


def tfidf_vec(train, test):
    tfidf_vec = TfidfVectorizer()
    train_vec = tfidf_vec.fit_transform(train)
    test_vec = tfidf_vec.transform(test)
    return train_vec, test_vec

def len_vec(train, test):
    train_len = np.zeros(len(train), dtype=np.int)
    for i, message in enumerate(train):
        train_len[i] = len(message)
    test_len = np.zeros(len(test), dtype=np.int)
    for i, message in enumerate(test):
        test_len[i] = len(message)
    return  train_len.reshape(-1,1), test_len.reshape(-1,1)


def punc_vec(train, test):
    punc_len_train = np.zeros(len(train), dtype=np.int)
    for i, message in enumerate(train):
        punc = [w for w in message if w in string.punctuation]
        punc_len_train[i] = len(punc)

    punc_len_test = np.zeros(len(test), dtype=np.int)
    for i, message in enumerate(test):
        punc = [w for w in message if w in string.punctuation]
        punc_len_test[i] = len(punc)

    return punc_len_train.reshape(-1, 1), test_len.reshape(-1, 1)



# define models

def MN_NB():
    clf = MultinomialNB()

    return clf

def SVM():
    clf = svm.SVC()

    return clf

def RF():
    parameters1 = {'n_estimators': [n for n in range(50, 300, 50)],
                   'criterion': ["gini", "entropy"],
                   'max_depth': (None, 4, 8, 12, 16, 20, 24, 50),
                   'min_samples_split': (2, 4, 6, 8, 10, 20, 30),
                   'min_samples_leaf': (16, 4, 12)}

    clf = GridSearchCV(RandomForestClassifier(),
                       parameters1,
                       cv=5,
                       n_jobs=-1,
                       scoring="accuracy")

    return clf


def xgb(X_train, y_train, X_test):
    num_of_runs = 10
    if not os.path.exists('optimization_result.csv'):
        os.mknod('optimization_result.csv')

    optimization_output_path = "optimization_result.csv"

    hyper_list = []
    n = 1
    for i in range(num_of_runs):
        print(f"Training model {n} out of {num_of_runs}")
        learning_rate = np.random.uniform(0.001, 0.15)
        max_depth = np.random.choice([3, 4, 5, 6])
        n_estimators = np.random.randint(low=50, high=180)
        subsample = min(np.random.uniform(0.6, 1.1), 1.0)
        colsample_bytree = min(np.random.uniform(0.6, 1.1), 1.0)

        params = {'learning_rate': learning_rate,
                  'max_depth': max_depth,
                  'n_estimators': n_estimators,
                  'subsample': subsample,
                  'colsample_bytree': colsample_bytree}
        print(params)

        clf = XGBClassifier(learning_rate=learning_rate,
                            objective='binary:logistic',
                            random_state=42,
                            n_jobs=8,
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree)

        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        print(classification_report(y_test, preds))
        acc = accuracy_score(y_test, preds)
        params['acc'] = acc

        hyper_list.append(pd.DataFrame(params, index=[0]))
        n = n + 1

    hyper_df = pd.concat(hyper_list)
    hyper_df.sort_values('acc', inplace=True, ascending=False)
    hyper_df.reset_index(drop=True, inplace=True)
    hyper_df.to_csv(optimization_output_path)

    best_clf = XGBClassifier(learning_rate=hyper_df['learning_rate'][0],
                             objective='binary:logistic',
                             random_state=42,
                             n_jobs=8,
                             n_estimators=hyper_df['n_estimators'][0],
                             max_depth=hyper_df['max_depth'][0],
                             subsample=hyper_df['subsample'][0],
                             colsample_bytree=hyper_df['colsample_bytree'][0])

    return best_clf


# fit and evaluate model
def fit_eval(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print(f"accuracy score: {accuracy_score(y_test, preds)}")
    print("\n=========================\n")


train_count, test_count = count_vec(train[0], test[0])
train_tf, test_tf = tfidf_vec(train[0], test[0])
train_len, test_len = len_vec(train[0], test[0])
punc_len_train, punc_len_test = punc_vec(train[0], test[0])

"""### Train and evaluate models"""

naive_bayes = MN_NB()
print("-----Model: Naive Bayes----")
print("CountVectorizer:")
fit_eval(naive_bayes, train_count, y_train, test_count, y_test)
print("TfidfVectorizer:")
fit_eval(naive_bayes, train_tf, y_train, test_tf, y_test)
print("LengthVector:")
fit_eval(naive_bayes, train_len, y_train, test_len, y_test)
print("PuncVector")
fit_eval(naive_bayes, punc_len_train, y_train, punc_len_test, y_test)

random_forest = RF()
print("-----Model: RandomForest ----")
print("CountVectorizer:")
fit_eval(random_forest, train_count, y_train, test_count, y_test)
print("TfidfVectorizer:")
fit_eval(random_forest, train_tf, y_train, test_tf, y_test)
print("LengthVector:")
fit_eval(random_forest, train_len, y_train, test_len, y_test)
print("PuncVector:")
fit_eval(random_forest, punc_len_train, y_train, punc_len_test, y_test)

xgb = xgb(train_count, y_train, test_count)
print("-----Model: xgboost ----")
print("CountVectorizer:")
fit_eval(xgb, train_count, y_train, test_count, y_test)
xgb = xgb(train_tf, y_train, test_tf)
print("CountVectorizer:")
fit_eval(xgb, train_tf, y_train, test_tf, y_test)
print("LengthVector:")
fit_eval(xgb, train_len, y_train, test_len, y_test)
print("PuncVector:")
fit_eval(xgb, punc_len_train, y_train, punc_len_test, y_test)


svm = SVM()
print("-----Model: SVM ----")
print("CountVectorizer:")
fit_eval(svm, train_count, y_train, test_count, y_test)
print("TfidfVectorizer:")
fit_eval(svm, train_tf, y_train, test_tf, y_test)
print("LengthVector:")
fit_eval(svm, train_len, y_train, test_len, y_test)
print("PuncVector:")
fit_eval(svm, punc_len_train, y_train, punc_len_test, y_test)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", type=str, default="SMSSpamCollection.txt",help="SMSSpamCollection data")

    args = parser.parse_args()

    main(args.datafile)