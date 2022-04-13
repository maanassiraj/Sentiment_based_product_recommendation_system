import nltk
nltk.download("stopwords")
nltk.download("punkt")
import pandas as pd
import numpy as np
import re
import string
import spacy
import pickle
import requests
from io import BytesIO
import random
random.seed(100)
np.random.seed(100)


def pickle_load(link):
    with BytesIO(requests.get(link).content) as f:
        return pickle.load(f)


def data_load():
    dataset = pd.read_csv("https://raw.githubusercontent.com/maanassiraj/Sentiment_based_product_recommendation_system/master/sample30.csv")
    data = dataset[["name", "reviews_text", "reviews_title"]]
    data.drop_duplicates(inplace=True)
    data = data.loc[~(data["name"].isnull() | data["reviews_text"].isnull() | data["reviews_title"].isnull())]
    data["reviews_text"] = data["reviews_text"] + " " + data["reviews_title"]
    data.drop(columns= "reviews_title", inplace= True)
    return data



def text_preprocessing_1(review):
    unwanted_text = "this review was collected as part of a promotion."
    review = review.lower().replace(unwanted_text, "").strip()
    review = re.sub("\.{2,}", " ", review)
    review = re.sub("[^a-z'\s]", "", review)
    for ch in string.ascii_lowercase:
        if ch != "l":
            review = re.sub(f"{ch}" + "{3,}", f"{ch}", review)
        else:
            review = re.sub(f"{ch}" + "{3,}", f"{ch}" + f"{ch}", review)
    return review



def text_preprocessing_2(reviews, nlp):
    return [" ".join([token.lemma_ for token in review if token.pos_ not in ["PROPN", "NOUN"]]) for review in nlp.pipe(reviews)]

  
def text_preprocessing_3(review):
    stop_words = [word for word in nltk.corpus.stopwords.words("english") if word not in ["don't", "doesn't", "do", "not", "did"]]
    review = " ".join([word for word in nltk.tokenize.word_tokenize(review) if word not in stop_words])
    review = review.replace("-PRON-", "")
    return review


def generate_perc_pos_reviews(prod_name, data, nlp, tf_idf_vectorizer, sent_model):
    prod_reviews = data.loc[data["name"] == prod_name, "reviews_text"]
    text_preproc_1 = np.vectorize(text_preprocessing_1)
    prod_reviews = pd.Series(text_preproc_1(prod_reviews))
    prod_reviews = pd.Series(text_preprocessing_2(prod_reviews, nlp))
    text_preproc_3 = np.vectorize(text_preprocessing_3)
    prod_reviews = pd.Series(text_preproc_3(prod_reviews))
    prod_reviews.drop_duplicates(inplace= True)
    prod_reviews = tf_idf_vectorizer.transform(prod_reviews)
    predictions = sent_model.predict(prod_reviews)
    return np.sum(predictions) / len(predictions)


    
def generate_top5_prod_recom(user_name):
    nlp = spacy.load("en_core_web_sm", disable = ["parser", "ner"])
    data = data_load()
    recommendation_eng = pickle_load("https://github.com/maanassiraj/Sentiment_based_product_recommendation_system/blob/master/pickle_files/recommendation_engine.pkl?raw=true")
    predicted_rat = recommendation_eng.loc[:, user_name].sort_values(ascending= False)
    top_20_recom = predicted_rat[predicted_rat > 0].iloc[:20].index
    recom = pd.DataFrame({"prod_recommendations": top_20_recom})
    tf_idf_vectorizer = pickle_load("https://github.com/maanassiraj/Sentiment_based_product_recommendation_system/blob/master/pickle_files/tf_idf_vectorizer.pkl?raw=true")
    sent_model = pickle_load("https://github.com/maanassiraj/Sentiment_based_product_recommendation_system/blob/master/pickle_files/sentiment_model.pkl?raw=true")
    recom["perc_pos_reviews"] = recom["prod_recommendations"].apply(generate_perc_pos_reviews, args=(data, nlp, tf_idf_vectorizer, sent_model))
    return list(recom.sort_values(by="perc_pos_reviews", ascending=False)["prod_recommendations"].iloc[:5])