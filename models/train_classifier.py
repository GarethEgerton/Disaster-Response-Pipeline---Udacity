

import pickle
import re
import sys
import time
import dill

import nltk
import numpy as np
import pandas as pd
import sqlalchemy
from joblib import dump, load
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, recall_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     train_test_split)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords',
               'maxent_ne_chunker', 'words', 'omw'])

run = "python train_classifier.py ../data/Disaster_Msgs.db classifier.pkl"


def load_data(database_filepath):
    '''Load SQL database, returning DataFrame and X data and Y labels '''
    path = 'sqlite:///' + database_filepath
    engine = create_engine(path)
    df = pd.read_sql('SELECT * FROM df', engine)
    X = df['message'].values
    Y = df.iloc[:, 4:].values
    return X, Y, df


def tokenize(text):
    ''''Remove punctuation and stopwords, normalise to lowercase and
    lemmatize'''
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text).lower())
    tokens = [token for token in tokens if token not in
              stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    return lemmed


def build_model():
    '''Build pipeline. Utilises CountVectorizer, Ttidf and Adaboost
    Classifier'''
    pipeline = Pipeline([
                       ('vect', CountVectorizer(tokenizer=tokenize)),
                       ('tfidf', TfidfTransformer()),
                       ('clf', MultiOutputClassifier(AdaBoostClassifier(),
                        n_jobs=-1))
                        ])
    return pipeline


def evaluate_model(model, X_test, Y_test, df):
    '''Predict Y-test and return average precision, recall and F1-score'''
    start_time = time.time()
    y_pred = model.predict(X_test)
    print("Prediction time--- %s seconds ---" % (time.time() - start_time))

    y_test_df = pd.DataFrame(Y_test)
    y_pred_df = pd.DataFrame(y_pred)

    # create list of precision, recall and f1-score per category
    scores = []
    for column in y_pred_df.columns:
        scores.append(precision_recall_fscore_support(y_test_df[column],
                                                      y_pred_df[column],
                                                      average='weighted'))

    # add column metric labels and category indexes to results dataframe
    results_df = pd.DataFrame(scores, columns=['precision', 'recall',
                                               'f1-score', 'drop'],
                              index=df.iloc[:, 4:].columns).drop('drop',
                                                                 axis=1)

    # print average column result metrics across all categories
    print(results_df.mean())


def save_model(model, model_filepath):
    '''Save trained model as pickle file'''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, df = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2)
        print('Building model...')
        model = build_model()
        print('Training model...')
        start_time = time.time()
        model.fit(X_train, Y_train)
        print("Training time --- %s seconds ---" % (time.time() - start_time))
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, df)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
    return model



if __name__ == '__main__':
    model = main()

with open('classifierX.pkl', 'wb') as file:
    dill.dump(model, file)
