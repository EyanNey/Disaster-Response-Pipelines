# **Project :** ***Disaster Response Pipelines***
![Headline](images/Disaster%20Response%20Pipelines%20Project%20-%20Headline.png)

## Table of Contents
1. **Description**
2. **Getting Started**
   - *Project Details*
   - *Dependencies*
   - *Program Execution*
3. **User Interface Display**
4. **Licensing, Authors, and Acknowledgements**


### 1. Description
This is an Udacity Data Scientist Nanodegree Program's project - "Disaster Response Pipelines" in collaboration with Figure Eight.
The main objective of this project is to create a Machine Learning Pipeline to categorize real-life disaster events and eventually send it to appropriate disaster relief agency.


### 2. Getting Started
#### **a. Project Details :** 
- This project is divided into 3 sections below :
    1. **ETL Pipeline** 
       - **Extract :** *To extract data from sources*
       - **Transform :** *To transform/cleaning the data*
       - **Load :** *To load/save the data into databse structure*
    2. **Machine Learning Pipeline**
       - *To build a Text Processing and Machine Learning Pipeline that able to classify text messages into categories*
    3. **Flask Web App**
       - *To classify the disaster events in real time*
       - *Dependencies*
       - *Program Execution*
- Project's programs are available in this Github respository [Disaster Response Pipelines](https://github.com/EyanNey/Disaster-Response-Pipelines)
#### **b. Dependencies :** 
  - *Python 3.5+*
  - Machine Learning Libraries : *NumPy, SciPy, Pandas, Sciki-Learn*
  - Natural Language Process Libraries : *NLTK*
  - SQLlite Database Libraqries: *SQLalchemy*
  - Web App and Data Visualization : *Flask, Plotly*
#### **c. Program Execution :** 
1. Run the following commands in the project's root directory to set up your database and model
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
![ETL Pipeline](images/ETL%20Pipeline.png)
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
![ML Pipeline](images/ML%20Pipeline.png)
2. Run the following command in the app's directory to run your web app
    `python app/run.py`
![Flask Web App](images/Flask%20Web%20App.png)
3. Go to http://0.0.0.0:3001/
![Layout](images/Disaster%20Response%20Pipelines%20Project%20-%20Headline.png)


### 3. User Interface Display
  - *On the Web main page, type in the sample message and click **"Classify Message"** to test on the model performance* 
![Sample Message](images/Sample%20Message.png)
  - *After clicking **"Classify Message"**, those relevant categories will be highlighted in green as shown below*
![Message Categories](images/Message%20Categories.png)


### 4. Licensing, Authors, Acknowledgements
1. Acknowledgements to 
   - Udacity Data Scientist Nano Degree courses for some of code ideas
   - Figure Eight for providing real-life disaster events to train my model 
2. Otherwise, feel free to use the code here as you would like !
