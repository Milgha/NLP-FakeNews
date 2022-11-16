# text preprocessing modules
from string import punctuation
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI 

app = FastAPI(
    title="Sentiment Model API",
    description="A simple API that use NLP model to predict the sentiment of the movie's reviews",
    version="0.1",
)

# load the sentiment model
with open(
    join(dirname(realpath('__file__')), "sentiment_model_pipeline.pkl"), "rb"
) as f:
    model = joblib.load(f)


# cleaning the data
Extended_Stop_Words = stopwords.words('english')
Extended_Stop_Words.extend(['elon','musk','twitter'])
    
def clean_tweet(tweet):
#    if type(tweet) == np.float:
#        return ""
    temp = tweet.lower() # lower case the text
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp) # removes mentions
    temp = re.sub("#[A-Za-z0-9_]+","", temp) # removes hashtags
    temp = re.sub('[0-9]+', '', temp)
    temp = re.sub(r'http\S+', '', temp) # removes URL's
    temp = re.sub('[()!?]', ' ', temp) # removes special charecters
    temp = re.sub('\[.*?\]',' ', temp) # further removes special charecters
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split() # splitting the words
    
    lemmatizer = WordNetLemmatizer()
    temp = [lemmatizer.lemmatize(w) for w in temp if not w in Extended_Stop_Words] # Removes the stop words and Lemming
    temp = " ".join(word for word in temp) # join all words together
    return temp

@app.get("/predict-review")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    # clean the review
    cleaned_review = clean_tweet(review)
    
    # perform prediction
    prediction = model.predict([cleaned_review])
    output = prediction[0]
#    probas = model.predict_proba([cleaned_review])
#    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    sentiments = {'Negative': "Negative", 'Positive': "Positive", 'Neutral': "Neutral"}
    
    # show results
    result = {"prediction": sentiments[output]}
    return result
