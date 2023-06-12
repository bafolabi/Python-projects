from flask import Flask, request, render_template, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from string import punctuation
import re, os, random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
nltk.download("omw-1.4")
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import pickle
#import StringIO
#import csv
from flask import make_response

nltk.download('stopwords')

set(stopwords.words('english'))

app = Flask(__name__)
pred= None
@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    final = False

    label = {0: "negative", 1: "positive"}

    # Content validation
    text = request.form['text1'].lower()

    if text == "" or text == None:
        final = False
        return render_template('form.html', final=final)

    else:
        data = [clean_text(text)]
        prediction = inference(data)
        output = f'"{text}" is {label[int(prediction[0])]}'

        return render_template('form.html', final=True, is_file=False, result=output)
    
@app.route('/upload', methods = ['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        data = pd.read_csv(f)

        #Extract the sentiment and sentence columns
        text_data = []
        for data in data["text"]:
            if isinstance(data, str):
                text_data.append(clean_text(data))
            else:
                text_data.append('')

        prediction = inference(text_data)
        
        # classes = list(set(prediction))
        pos = round(len([a for a in prediction if int(a) == 1]) / len(prediction)*100,2)
        neg = round(len([a for a in prediction if int(a) == 0]) / len(prediction)*100, 2)
        
      #  data["prediction"] = data["text"] #list(prediction)
        data = pd.DataFrame({'text': text_data,'prediction': prediction})

        if os.path.exists("temp/prediction.csv"):
            os.remove("temp/prediction.csv")
        data.to_csv("temp/prediction.csv")

        return render_template('form.html', final=True, is_file=True, result=str(prediction), positive=pos, negative=neg, data=data)
    

def inference(data):

    model_pickle = pickle.load(open('model.pkl','rb'))
    Tfidf_pickle = pickle.load(open('vectorizer.pkl','rb'))
    label_enc = pickle.load(open('label_enc.pkl','rb'))

    lemmatizer = WordNetLemmatizer()
    lemmatized_messages = []

    for message in data:
        lemmatized_message = " ".join([lemmatizer.lemmatize(word,pos="v") for word in message.split()])
        lemmatized_messages.append(lemmatized_message)


    X = Tfidf_pickle.transform(lemmatized_messages)

    y = model_pickle.predict(X)
    #y = label_enc.inverse_transform(y)

    return y
    
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers["X-UA-Compatible"] = "IE=Edge,chrome=1"
    response.headers["Cache-Control"] = "public, max-age=0"
    return response

@app.route('/download')
def download_csv():
    cache_buster = random.randint(0, 10000)
    return send_file('temp/prediction.csv',
                     mimetype='text/csv',
                     attachment_filename=f'report_{cache_buster}.csv',
                     as_attachment=True)

#data preprocessing 
def clean_text(text):
    # Remove Twitter #tags and @usernames
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
