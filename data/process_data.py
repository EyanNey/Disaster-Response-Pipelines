import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load Data function:
    Load messages and categories files, merge and store as DataFrame
    
    Arguments:
        messages_filepath -> filepath for disaster_messages.csv file
        categories_filepath -> filepath for disaster_categories.csv file
    Output:
        df -> Loaded, merge and store csv file as Pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'left',\
                    on=['id'])
    return df 


def clean_data(df):
    """
    Clean Data function
    
    Arguments:
        df -> Raw data Pandas DataFrame
    Outputs:
        df -> Clean data Pandas DataFrame
    """
    categories = df['categories'].str.split(";", expand=True)
    firstrow = categories.iloc[0]
    category_colnames = firstrow.apply(lambda x:x.split('-')[0])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])        
        # assert that all columns have values 0 or 1
        categories[column] = categories[column].apply(lambda x:1 if x>1 else x)        
        
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)        
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)        
    # drop duplicates
    df.drop_duplicates(inplace=True)        
    # remove nans based on original column (high number of nans)
    df = df[~(df.isnull().any(axis=1))|(df.original.isnull())]
    return df


def save_data(df, database_filename):
    """
    Save Data function
    
    Arguments:
        df -> Clean data Pandas DataFrame
        database_filename -> Database file (.db) destination path
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Disaster_Response_ETL', engine, if_exists='replace', index=False)
    pass  


def main():
    """
    Main Data Processing function
    
    This function implement the ETL pipeline:
        1) Data extraction from .csv
        2) Data cleaning and pre-processing
        3) Data loading to SQLite database
    """
    print(sys.argv)
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

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