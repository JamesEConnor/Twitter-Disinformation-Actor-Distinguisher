# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:27:18 2020

@author: James
"""

import numpy as np;
import pandas as pd;
import random;
import pickle;

from textblob import TextBlob;

from gensim.models import KeyedVectors;
from sklearn.decomposition import LatentDirichletAllocation;
from sklearn.feature_extraction.text import CountVectorizer;

from sklearn.cluster import OPTICS;

# Load a Word2Vec model
model = KeyedVectors.load_word2vec_format("topics/GoogleNews-vectors-negative300.bin", binary=True, limit=500000);

# Transform topics into vectors and return the result.
def quantify_topics(topics, topic_count=3):
    result = [];
    for topic_list in topics:
        counter = 0;
        quantified = [];
        for topic in topic_list:
            if counter >= 3:
                break;
                
            try:
                topic = TextBlob(topic).translate().text;
            except:
                pass;
                
            try:
                quantified.append(model[topic].tolist());
                counter += 1;
            except:
                continue;
                
        if len(quantified) == 0:
            result.append([[0] * 300]);
        else:
            result.append(quantified + ([0] * (topic_count - len(quantified))));
    
    return result;

# Show top n keywords for each topic
def show_topics(vectorizer, lda_model, n_words=3):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
        
    return topic_keywords

# Get the topics of a number of tweets as words.
def get_topics(tweets, n_topics=3):
    n_tweets = len(tweets);
    if len(tweets) == 1:
        tweets = tweets[0].split(' ');
    
    # LDA can only use raw term counts for LDA because it is a probabilistic graphical model
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=1, max_features=100, stop_words='english');
    tf = tf_vectorizer.fit_transform(tweets);
    lda = LatentDirichletAllocation(n_components=n_tweets, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf);
    return show_topics(tf_vectorizer, lda, n_words=n_topics);
 
# Get the topics of a number of tweets as word vectors.
def get_topics_quantified(tweets, n_topics=3):
    topics = get_topics(tweets, 2*n_topics);
    # print(topics);
    return quantify_topics(topics, topic_count=n_topics);

# Take in tweets and cluster them using DBSCAN
def cluster_topics(tweets, model):
    quantified = get_topics_quantified(tweets, n_topics=1);
    quantified = [x[0] for x in quantified];
    
    file = open(model, 'rb');
    optics = pickle.load(file);
    # print(optics.labels_);
    # print("MAX: " + str(max(optics.labels_)));
    
    if len(tweets) < optics.get_params()["min_samples"]:
        optics.set_params(min_samples=len(tweets));
        
    # print(quantified);
    if len(tweets) != 1:
        return optics.fit_predict(quantified);
    else:
        quantified.append([0] * 300);
        result = optics.fit_predict(quantified);
        return [result[0]];

def train_topics(tweets, model, _min_samples=5):
    quantified = get_topics_quantified(tweets, n_topics=1);
    quantified = [x[0] for x in quantified];
    
    optics = OPTICS(min_samples=_min_samples);
    optics.fit(quantified);
    
    file = open(model, 'wb');
    pickle.dump(optics, file);
    file.close();
    



### CREATE TWEET ARRAY EXAMPLES
# tweets = ["Commended for no longer saying 'China virus' Did US military bring #Covid19 to 7th Military World Games Oct18-27, 2019 Wuhan, China? Patient zero: Maatja Benassi US Athlete/Intelligence Officer? Did World's military take it back to their countries?", "BREAKING | Boris Johnson will get lung ventilation - health source sptnkne.ws/BWtv #SputnikBreaking @BorisJohnson"]
# #tweets = ["BREAKING | Boris Johnson will get lung ventilation - health source sptnkne.ws/BWtv #SputnikBreaking @BorisJohnson"]
# #tweets = ["Commended for no longer saying 'China virus' Did US military bring #Covid19 to 7th Military World Games Oct18-27, 2019 Wuhan, China? Patient zero: Maatja Benassi US Athlete/Intelligence Officer? Did World's military take it back to their countries?"]
# tweets = ["The legislative council belongs to the people of Hong Kong.Those people with ulterior motives indicated by forces hide behind the scenes laid seige to the legislative.The path of your darkness and the bright roads of the masses of the Hong Kong people will not inevitably coexist."]

### GET AND PRINT TOPICS
# topics = get_topics(tweets);
# print(topics);

### GET AND PRINT TOPIC VECTORS
# quantified = get_topics_quantified(tweets);
# print(quantified);

### CLUSTER TWEETS BASED ON TOPICS
# print(train_topics(tweets, "models/model_001.pickle", _min_samples=1));
# print(cluster_topics(tweets, "models/model_001.pickle"));
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# # Read the CSV file mapping all tweet data to a motive.
# labeledDataPath = "../data/actors_and_motives.csv";
# df = pd.read_csv(labeledDataPath, usecols=["tweet_docs", "motive"], converters={"tweet_docs": lambda x: x.strip("[]").split(", ")});

# # Removes leading and ending quote characters
# df["tweet_docs"] = [[x.strip('\"') for x in df["tweet_docs"][i]] for i in range(len(df["tweet_docs"]))];

# MAX_LINES = 30;
# tweets = [];

# # Get all file paths and their associated motives from the dataframe.
# _files, _classes = [], [];
# for i in range(len(df["tweet_docs"])):
#     for j in range(len(df["tweet_docs"][i])):
#         data = pd.read_csv(df["tweet_docs"][i][j], usecols=["tweet_text"], nrows=MAX_LINES);
#         tweets += data["tweet_text"].tolist();
        
# train_topics(tweets, "models/model_001.pickle");


# tweets = ["Commended for no longer saying 'China virus' Did US military bring #Covid19 to 7th Military World Games Oct18-27, 2019 Wuhan, China? Patient zero: Maatja Benassi US Athlete/Intelligence Officer? Did World's military take it back to their countries?", "BREAKING | Boris Johnson will get lung ventilation - health source sptnkne.ws/BWtv #SputnikBreaking @BorisJohnson"]
# labels = cluster_topics(tweets, "models/model_001.pickle");
# connected = sorted(zip(labels, tweets));
# for label, tweet in connected:
#     print(str(label) + ": " + str(tweet));