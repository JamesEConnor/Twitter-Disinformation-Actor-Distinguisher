# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:17:16 2020

@author: James
"""

import pickle;
import pandas as pd;

from sklearn.neighbors import KNeighborsClassifier;
from sklearn import metrics;

from train_actor_clustering import combine_measures;

# NEED:
# tweet_text
# account_creation_date
# account_language
# tweet_language
# tweet_time
# is_retweet

highestCorrectDistance = -1;
totalCorrect = {};
totalChecked = 0;

def get_actors(df, model):
    file = open(model, 'rb');
    knn = pickle.load(file);
    file.close();
    
    df = combine_measures(df);
    
    distance, index = knn.kneighbors(df);
    return knn.predict(df);

def get_actor(tweet_text, account_creation_date, account_language, tweet_language, tweet_time, is_retweet, correct=None):
    df = pd.DataFrame(data=[[tweet_text, account_creation_date, account_language, tweet_language, tweet_time, is_retweet]],
            columns=["tweet_text", "account_creation_date", "account_language", "tweet_language", "tweet_time", "is_retweet"]);
    
    prediction = get_actors(df, "actors_model_k1.pickle")[0];










# print("\nCORRECT LABEL: prc\nPREDICTED LABEL: " + str(get_actor("Commended for no longer saying 'China virus' Did US military bring #Covid19 to 7th Military World Games Oct18-27, 2019 Wuhan, China? Patient zero: Maatja Benassi US Athlete/Intelligence Officer? Did World's military take it back to their countries?", "2009-07-17", "en", "en", "2020-03-27 20:27", "false")));
    
# Read the CSV file mapping all tweet data to a motive.
labeledDataPath = "data/actors_and_motives.csv";
df = pd.read_csv(labeledDataPath, usecols=["apm", "tweet_docs"], converters={"tweet_docs": lambda x: x.strip("[]").split(", ")});

# Removes leading and ending quote characters
df["tweet_docs"] = [[x.strip('\"') for x in df["tweet_docs"][i]] for i in range(len(df["tweet_docs"]))];

# Get all file paths and their associated motives from the dataframe.
_files, _classes = ([], []);
for i in range(len(df["tweet_docs"])):
    for j in range(len(df["tweet_docs"][i])):
        _files.append(df["tweet_docs"][i][j]);
        _classes.append(df["apm"][i]);

finalPredictions = [];
finalResults = [];
for index in range(len(_files)):
    print(str(index + 1) + "/" + str(len(_files)));
    try:
        dataframe = pd.read_csv(_files[index], usecols=["tweet_text", "account_creation_date", "account_language", "tweet_language", "tweet_time", "is_retweet"], nrows=10, header=0);
        
        predicted = get_actors(dataframe, 'models/sampled_200_actors_model_k1.pickle');

                
        finalPredictions.extend(predicted);
        
        finalResults.extend([_classes[index]] * len(predicted));
        
        # 100           0.831973898858075
        # 120           0.8482871125611745
        # 200           0.8858075040783034
        # 340           0.8189233278955954
        # 600           0.8172920065252854
    except:
        continue;
        
print(metrics.accuracy_score(finalPredictions, finalResults));
    
print(highestCorrectDistance);