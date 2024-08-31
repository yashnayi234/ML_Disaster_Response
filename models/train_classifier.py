import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import re
import numpy as np

from sklearn import multioutput
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

from startverbex import StartingVerbExtractor

import matplotlib.pyplot as plt
# %matplotlib inline

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import os
import pickle


nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.data.path.append(os.getcwd())
nltk.download('averaged_perceptron_tagger', download_dir=os.getcwd())
    
    
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

    
def load_data(database_filepath):
    '''
     Load data from the SQLite database.

    Parameters:
    database_filepath: The file path for the SQLite database.

    Returns:
    X: Features dataset (messages).
    y: Labels dataset (categories).
    y_cols: List of category names.
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('MessagesCategories', engine)
    y_cols = []

    for i in df.columns:
        if i not in ['id', 'message', 'original', 'genre']:
            y_cols.append(i)
    
    X = df['message']
    y = df[y_cols]
    
    return X,y,y_cols


def tokenize(text):
    '''
        Input: take the raw text
        Process: normalized, stopwords removed, stemmed and lemmetized 
        
        Output: return a list of lemmatized tokens.
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
   
    pipeline = Pipeline([
        ('feature', FeatureUnion([
            ('test_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf',TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters ={'clf__estimator__n_estimators': [10,20],
                'clf__estimator__min_samples_split': [2, 3]
                }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=2, n_jobs=1)
    return cv

def ML_classification_report(y_test, y_pred):
    '''
        INPUT: y_test, y_pred
        
        PROCESS: Generate Classification Report by takeing precision, recall, f1, support, accuracy
        
        OUTPUT: DataFrame of precision, recall, f1-score, support, and accuracy
    
    '''
    
    report = {}
    for i, column in enumerate(y_test.columns):
        # Calculate precision, recall, and f1-score for each column
        precision = precision_score(y_test[column], y_pred[:, i], average='weighted')
        recall = recall_score(y_test[column], y_pred[:, i], average='weighted')
        f1 = f1_score(y_test[column], y_pred[:, i], average='weighted')
        support = y_test[column].sum()
        accuracy = accuracy_score(y_test.iloc[:, i].values, y_pred[:, i])

        # Store the results in a dictionary
        report[column] = {
            'precision': precision.round(2),
            'recall': recall.round(2),
            'f1-score': f1.round(2),
            'support': support,
            'accuracy': accuracy.round(2)
        }
    
    # Convert the dictionary to a DataFrame and transpose it
    return report[column]


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(ML_classification_report(Y_test, y_pred))
    pass


def save_model(model, model_filepath):
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()