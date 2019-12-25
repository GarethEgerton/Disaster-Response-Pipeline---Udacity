import sys

import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Loads messages and categories CSV files and merges them into one
    dataframe'''
    messages = pd.read_csv(messages_filepath)
    cat = pd.read_csv(categories_filepath)
    df = messages.merge(cat, how='inner', on='id')
    return df


def clean_data(df):
    '''cleans data into ML ready format'''

    # split categories into separate columns
    categories = df.categories.str.split(';', expand=True)

    # select categories
    row = categories.iloc[0]

    # filtering for category column names
    categories.columns = row.apply(lambda x: x[:-2])

    # iterating and filtering for raw numerical data in each column
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])

    # droping old categories and replacing with new, removing duplicates
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''Saves data as SQL database'''
    sql_file_name = 'sqlite:///' + database_filename
    engine = create_engine(sql_file_name) 
    df.to_sql('df', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath =
        sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()