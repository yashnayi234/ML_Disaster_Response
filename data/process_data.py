import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge message and category datasets.

    Parameters:
    messages_filepath: The file path for the CSV file containing the messages.
    categories_filepath: The file path for the CSV file containing the categories.

    Returns:
    DataFrame: A DataFrame containing the merged data from both files, 
               with messages and their corresponding categories combined based on the 'id'.
    """
    message = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(message, categories, how='inner', on=['id'])
    return df


def clean_data(df):
    """
    Cleans the combined DataFrame by processing the categories, converting them into numerical values, 
    and removing duplicates.

    Parameters:
    df (DataFrame): The DataFrame containing the merged messages and categories data.

    Returns:
    DataFrame: A cleaned DataFrame with categories converted into separate binary columns and duplicates removed.
    """
    categories = df['categories'].str.split(pat=';',expand=True) 
    row = categories.loc[0,:]
    
    category_colnames = []
    for i in row:
        category_colnames.append(i[0:i.find('-')])
        
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[len(x)-1])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
        # converting number greater than 1 to 1
        categories[column] = categories[column].apply(lambda x: 1 if x > 1 else x)
        
    df=df.drop(['categories'], axis=1)
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df=df.drop_duplicates(subset=['message'], keep='first')
    return df


def save_data(df, database_filename):
    '''
    Saves the cleaned DataFrame into a SQLite database.

    Parameters:
    df (DataFrame): The DataFrame containing the cleaned data.
    database_filename (str): The filename for the SQLite database where the data will be saved.

    Returns:
    None
    '''
    engine = f'sqlite:///{database_filename}'
    df.to_sql('MessagesCategories', engine, index=False, if_exists='replace')

    


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print(sys.argv[1:])

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
