# ML Disaster Response Pipeline

## Project Overview
This project is part of the Data Scientist Nanodegree and focuses on building a machine learning pipeline to categorize disaster response messages. The goal is to facilitate communication with appropriate disaster relief agencies by classifying messages into multiple categories. The project includes an ETL pipeline, an ML pipeline, and a Flask web app for model deployment and data visualization.

## Project Components
1. **ETL Pipeline** (`data/process_data.py`)
    - **Load Data:** Messages and categories datasets.
    - **Clean Data:** Remove duplicates and process text.
    - **Store Data:** Save the cleaned data to an SQLite database.

2. **ML Pipeline** (`models/train_classifier.py`)
    - **Load Data:** Extract data from the SQLite database.
    - **Build Pipeline:** Text processing and classification using GridSearchCV for hyperparameter tuning.
    - **Evaluate Model:** Test set evaluation metrics.

3. **Flask Web App** (`app/run.py`)
    - **Input:** Users can input new messages for classification.
    - **Output:** The model classifies the message into categories and visualizes data.

## Installation Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yashnayi234/ML_Disaster_Response.git

Install Dependencies: Make sure you have Python 3.6+ installed. Install the required libraries:


```pip install -r requirements.txt```
Run ETL Pipeline:


```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
Train ML Model:


```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```
Run the Web App:

```python run.py```
