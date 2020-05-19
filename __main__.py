# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:17:16 2020

@author: James
"""

import pickle;
import pandas as pd;

from sklearn.neighbors import KNeighborsClassifier;

from train_actor_clustering import combine_measures;

# NEED:
# tweet_text
# account_creation_date
# account_language
# tweet_language
# tweet_time
# is_retweet

def get_actors(df, model):
    print("OPENING MODEL...");
    file = open(model, 'rb');
    knn = pickle.load(file);
    file.close();
    
    print("QUANTIFYING DATA...");
    df = combine_measures(df);
    print("PREDICTING...");
    return knn.predict(df);

def get_actor(tweet_text, account_creation_date, account_language, tweet_language, tweet_time, is_retweet):
    df = pd.DataFrame(data=[[tweet_text, account_creation_date, account_language, tweet_language, tweet_time, is_retweet]],
                 columns=["tweet_text", "account_creation_date", "account_language", "tweet_language", "tweet_time", "is_retweet"]);
    
    return get_actors(df, "actors_model_k1.pickle")[0];

print("\nCORRECT LABEL: prc\nPREDICTED LABEL: " + str(get_actor("Commended for no longer saying 'China virus' Did US military bring #Covid19 to 7th Military World Games Oct18-27, 2019 Wuhan, China? Patient zero: Maatja Benassi US Athlete/Intelligence Officer? Did World's military take it back to their countries?", "2009-07-17", "en", "en", "2020-03-27 20:27", "false")));