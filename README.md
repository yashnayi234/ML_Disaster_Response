# ML_Disaster_Response

# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

5. Disaster Response Pipeline
Data Scientist Nanodegree - Data Engineering
Table of Contents
Project Overview
Project Components
ETL Pipeline
ML Pipeline
Flask Web App
Instructions
Data Cleaning
Training Classifier
Starting the Web App
Files
Software Requirements
Credits and Acknowledgements
Project Overview
This project applies data engineering techniques to analyze disaster data from Figure Eight. The goal is to build a machine learning model that categorizes disaster-related messages, enabling effective communication with appropriate disaster relief agencies.

The dataset contains real messages sent during disaster events. This project includes a web app where emergency workers can input a new message and receive classification results in multiple categories. The web app also displays visualizations of the data.

Project Components
ETL Pipeline
The ETL pipeline in data/process_data.py performs the following tasks:

Loads the messages and categories datasets.
Merges the datasets.
Cleans the data.
Stores the cleaned data in an SQLite database.
ML Pipeline
The machine learning pipeline in models/train_classifier.py:

Loads data from the SQLite database.
Splits the data into training and testing sets.
Constructs a text processing and machine learning pipeline.
Trains and fine-tunes a model using GridSearchCV.
Evaluates the model's performance on the test set.
Flask Web App
The web app allows users to input a new message and receive classification results. It also displays data visualizations. The app is powered by Flask and is structured as follows:

app/run.py - The main Flask application file.
app/templates/ - HTML templates for the web pages.
Instructions
Data Cleaning
To clean the data, run:

bash
Copy code
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
Training Classifier
To train the classifier, run:

bash
Copy code
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Starting the Web App
To start the web app, run:

bash
Copy code
python app/run.py
Then, open http://localhost:3001 in your web browser.

Files
csharp
Copy code
.
├── app
│   ├── run.py              # Main Flask app
│   ├── static
│   │   └── favicon.ico     # Web app favicon
│   └── templates
│       ├── go.html         # Result page
│       └── master.html     # Main page
├── data
│   ├── DisasterResponse.db # SQLite database
│   ├── disaster_categories.csv # Original category data
│   ├── disaster_messages.csv # Original message data
│   └── process_data.py     # ETL pipeline script
├── img                     # Visualizations for README and web app
├── models
│   └── train_classifier.py # ML pipeline script
Software Requirements
This project uses Python 3.6.6. The required libraries are listed in requirements.txt. Additional standard libraries include collections, json, operator, pickle, pprint, re, sys, time, and warnings.

Credits and Acknowledgements
Thanks to Udacity for their support and for allowing the use of their logo as the favicon for this web app. Also, special thanks to this blog post for motivating improved documentation.


