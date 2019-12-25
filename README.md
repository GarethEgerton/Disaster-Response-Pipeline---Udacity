# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000/ 


### Summary:

-   Pre-labelled tweets and text messages have been provided by Figure 8. The data originally comes from communications across social media and messaging that have taken place following live disasters.
    
-   During and after a disaster there will typically be millions of tweets, txt messages and other social media messages sent back and forth. Unfortunately it is difficult to distinguish between which of these are critical and require immediate attention from response teams, and those which do not. Due to the sheer volume of messages, there is simply not enough time for human assesment of all communications at the time when it is needed most. Typically only 1 in 1000 messages actually requries direct attention. 
    
-   This project seeks to create a supervised machine learning model using natural language processing that can accurately categorise messages to be forwarded to relevant disaster response teams.

-   This is more sophisticated than simply using a keyword search which is unable to accurately capture the nuance in the text.

-   The focus is initally on cleaning the data and constructing an ETL pipeline. A ML pipeline is then constructed, maximising accuarcy with gridsearch paramter tuning. Finally the model is deployed to a Flask web app where new messages can be predicted and categorised. 


### Data cleaning process:

-   The data was originally provided as two separate csv files, one containng the list of raw messages and the other the message category data labels. 
-   The first step was to merge the two datasets using the unique reference ids. 
-   Message categories had to be extracted from one messy text field into separate labelled columns per category and one hot encoded to be ready for the machine learning process.
- Finally any duplicated data was removed and the cleaned dataset saved to an sql database. 


### Machine learning model:

-   A tokenize function was created to remove punctuation, stopwords, make lowercase and finally lematize tokens.
-   An ML pipeline was created to:
    1.  Tokenize
    2.  Perform tfidftranformation
    3.  Train using a multi-output adaboost classifier.
    4.  Gridsearch employed to optimise parameters.


### Class imbalance:

-   The total number of messages labelled as each category is as follows:

![image](https://user-images.githubusercontent.com/45258467/54295754-913e1580-45ab-11e9-9a10-2d3d00c9d497.png)

-   As can be seen, there is severe class imbalance with categories ranging from 20,000 instances down to zero for the 'child alone category'.

![image](https://user-images.githubusercontent.com/45258467/54816757-fc16dd00-4c8c-11e9-8d9b-e18c9ed4c302.png)

-   Since there are zero instances of the "child_alone" category throughout the whole dataset. the resulting scoring for precision, recall and f1-score for this category are 100%. The model, however has no way of identifying what a "child alone" message might look like and if any were introduced, it would be unable to predict them. Similarly for other categories where we have limited training data - e.g. "offer", "shops" and "tools". 

-   If we plot the categories in order of frequency vs the scores achieved we can see that on average scores are higher the less frequent this category of message appears in the dataset.

![image](https://user-images.githubusercontent.com/45258467/54813998-b9520680-4c86-11e9-8be8-8e8e7b8a06d5.png)

-   This can also be seen in the following view where the scores are plotted directly against the number of instances of each category label.. 

![image](https://user-images.githubusercontent.com/45258467/54814043-d090f400-4c86-11e9-9851-302a149d9722.png)

- Interesting to note is that the recall score is always slightly above the precision score. Taking each individual category, the   precision, recall and F1-score actually do not vary significantly and quite tightly follow the same pattern.

### Precision vs Recall:

- Certain categories of messages are more important than others and it is essential that they are not missed by disaster recovery teams in particular where time is of the essence and lives are at stake. These would include categories such as 'search and rescue', 'missing people', 'fire' and 'medical help'. For these categories we would place more importance on recall, ensuring that all such messages are captured and passed on to teams. It is acceptable to have more false positives in exchange for ensuring no critical messages are missed.

- On the other hand we would want a greater emphasis on precision for less urgent categories to minimise the drain on limited resources allocated to false positives that need to be investigated. 


### Files:

1.  **Dataset**
    - disaster_categories.csv
        - raw data labels
    - disaster_messages.csv
        - raw data csv messages

2.  **Exploratory data analysis and cleaning usign Jupyter Notebook.**
    - ETL Pipeline Preparation.ipynb
        - Step by step explores and cleans raw messages data from csv before loading into SQL database.
    - ML Pipeline Preparation.ipynb
        - Explores step by step creation of ML pipeline, optimizing results using gridsearch for precision, recall and F1-score.
        
        
3.  **Scripts**
    - process_data.py
        - Command line script cleans data and stores in database. Follows steps outlined in ETL Pipeline Preparation.ipynb.
    - train_classifier.py  
        - Command line script that trains classifier using optimised parameters.
        
        
4.  **Flask web app**
    - run.py
    - go.html
    - master.html   
    
    




