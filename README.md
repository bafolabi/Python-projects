# COVID-19 Vaccine Sentiment Classifier WebApp

A web interface that performs sentiment analysis using machine learning

#### Basic Features
* Remove stop words 
* Pre-process text (remove punctuation, lower case)
* Lemmatization of words
* Predict a tweet sentiment 
* Batch upload for prediction
* Report downloads


#### Sentiment Analysis
* Classify the Covid-19 vaccine tweet into positive or negative sentiment.


### Prerequisites

This app is built using **Python 3.9**
This app can be opened via Visual Studio application

### Run the code to install the required libraries for the application
pip install -r requirements.txt

## Start Service
Now, to start the application, do the following:

    python app.py

Server will start and  you can connect by opening the web browser and connecting via the URL below.

    http://127.0.0.1:5002/

### Single Inferencing
Copy a COVID-19 related tweet in the input area and click on submit button. The tweet sentiment will appear below.


### Batch Upload

In order to upload a file for sentiment analysis successfully, adhere to these steps:

1.  Verify that the file format is CSV and that the column with the tweets for analysis is labeled "text".
2.  Once the prediction is complete, a download button will appear at the top of the statistics table. Click on this button to   
    download a report.
3.  Open the report file that was downloaded. An extra column named "prediction" will be appended to the report.
4.  In the "prediction" column, sentiments will be labeled as follows: 0 = negative, 1 = positive.

