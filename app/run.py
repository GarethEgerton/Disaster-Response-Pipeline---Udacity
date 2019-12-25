import json
import plotly
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import dill

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram, Table

from sklearn.externals import joblib

import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords',
               'maxent_ne_chunker', 'words', 'omw'])


def tokenize(text):
    ''''Remove punctuation and stopwords, normalise to lowercase and
    lemmatize'''
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text).lower())
    tokens = [token for token in tokens if token not in
              stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(token) for token in tokens]
    return lemmed

app = Flask(__name__)

# load data

#engine = create_engine('sqlite:///..\data\Disaster_Msgs.db')
engine = create_engine('sqlite:///Disaster_Msgs.db')
df = pd.read_sql_table('df', engine)
results = pd.read_sql_table('results', engine)

# load model





# def tokenize(text):
#     '''tokenizes and lemmatizes user input text'''
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens

# model = joblib.load("../models/classifier.pkl")

# model = joblib.load("classifier.pkl")
with open("classifierX.pkl",'rb') as f:
    model = dill.load(f)

def random_message():
    '''return random message from database and corresponding string of
    category labels'''

    # select message from random row in dataframe
    random = np.random.randint(df.shape[0]+1)
    message = df.iloc[random, :].message

    # create categories as comma separated string of message category labels
    categories = ""
    for i, j in enumerate(df.iloc[random, 4:]):
        if j == 1:
            if not categories:
                categories += df.iloc[random, 4:].index[i]
            else:
                categories += ", "
                categories += df.iloc[random, 4:].index[i]
    if not categories:
        categories = "none"
    return [message, categories]


def generate_messages(number):
    ''' return list of specified number of random_messages '''
    random_message_list = []
    for num in range(number):
        random_message_list.append(random_message())
    return random_message_list

random_message_list = generate_messages(10)


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    '''Extracts data from dataframe and creates plotly visuals'''

    # extract data needed for visuals
    # counts per genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # message character length distribution
    lengths = df['message'].apply(lambda x: len(x))

    # sum of message totals per category
    categories = df.iloc[:, 4:].columns.tolist()
    totals = df.iloc[:, 4:].sum().tolist()
    total_per_category = pd.DataFrame({'Categories': categories,
                                       'Totals': totals}).sort_values(
                                           by=['Totals'], ascending=False)

    # create plotly visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Histogram(
                    x=lengths
                )
            ],

            'layout': {
                'title': 'Distribution of Message Character Lengths',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'range': [0, 500],
                    'title': "Character Length"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=total_per_category.Categories,
                    y=total_per_category.Totals
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Table(
                    columnorder=[1, 2, 3, 4, 5],
                    columnwidth=[200, 100, 100, 100, 150],
                    header=dict(values=list(results.columns),
                                fill=dict(color='#C2D4FF'),
                                align=['center'] * 5),
                    cells=dict(values=[results['Categories'], results.precision,
                                       results.recall, results['f1-score'], 
                                       results['occurences']],
                               fill=dict(color='#F5F8FF'),
                               align=['center'] * 5))
            ],

            'layout': {
                'title': 'Model prediction scores per category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'width': 700,
                'height': 950
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON,
                           random_message_list=random_message_list)


# web page handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


# @app.route('/about')
# def about():
#     return render_template('about.html', ids=ids, graphJSON=graphJSON)

@app.route('/about')
def about():

    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graphs = [
        
        {
            'data': [
                Table(
                    columnorder=[1, 2, 3, 4],
                    columnwidth=[200, 100, 100, 100],
                    header=dict(values=list(results.columns),
                                fill=dict(color='#C2D4FF'),
                                align=['center'] * 5),
                    cells=dict(values=[results['index'], results.precision,
                                       results.recall, results['f1-score']],
                               fill=dict(color='#F5F8FF'),
                               align=['center'] * 5))
            ],

            'layout': {
                'title': 'Model prediction scores per category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'width': 700,
                'height': 950
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('about.html', ids=ids, graphJSON=graphJSON)

if __name__ == '__main__':
    app.run(debug=True)