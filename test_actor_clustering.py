# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:19:52 2020

@author: James
"""

import pandas as pd;
import numpy as np;
import pickle;
import math;

from motives.run_motive_classifier import classifyDF;
from topics.run_topic_quantifier import cluster_topics;
from methods.run_method_encoder import tweet_info;

from sklearn.neighbors import KNeighborsClassifier;
from sklearn import metrics;

# Combine all three measurements for KNeighborsClassifier
def combine_measures(dataframe): 
    methods_encoded = tweet_info(dataframe);   
    motive_classes = classifyDF(dataframe, "motives/models/model.pickle");
    topics_quantified = cluster_topics(dataframe["tweet_text"].tolist(), "topics/models/model_001.pickle");
    
    result = [];
    for index in range(len(motive_classes)):
        append = motive_classes[index];
        append.extend([topics_quantified[index]]);
        append.extend(methods_encoded[index]);
        result.append(append);
        
    return np.asarray(result);

def isolate_data_points(path):
    dataframe = pd.read_csv(path + ".csv");
    tweet_ids = dataframe["tweetid"].tolist();
    
    
    
    labeledDataPath = "data/actors_and_motives.csv";
    df = pd.read_csv(labeledDataPath, usecols=["tweet_docs", "apm"], converters={"tweet_docs": lambda x: x.strip("[]").split(", ")});
    
    # Removes leading and ending quote characters
    df["tweet_docs"] = [[x.strip('\"') for x in df["tweet_docs"][i]] for i in range(len(df["tweet_docs"]))];
    
    
    MAX_LINES = 300;
    first_file = True;
    index_counter = 0;
    
    with open(path + "_ISOLATE.csv", 'w', encoding="utf8") as file:
        for i in range(len(df["tweet_docs"])):
            for j in range(len(df["tweet_docs"][i])):        
                read_dataframe = pd.read_csv(df["tweet_docs"][i][j], nrows=MAX_LINES);
                
                with open(df["tweet_docs"][i][j], 'r', encoding="utf8") as twitter_file:
                    counter = -1;
                    for line in twitter_file:
                        if counter == -1:
                            if first_file is True:
                                file.write(line.rstrip() + ',"apm"\n');
                                first_file = False;
                        else:
                            if counter >= MAX_LINES:
                                break;
                            
                            if read_dataframe["tweetid"][counter] in tweet_ids:
                                file.write(dataframe_row_to_string(read_dataframe.iloc[counter], read_dataframe, df["apm"][i]) + "\n");
                                index_counter += 1;
                        
                        counter += 1;
      
cols = ["tweetid","userid","user_display_name","user_screen_name","user_reported_location","user_profile_description","user_profile_url","follower_count","following_count","account_creation_date","account_language","tweet_language","tweet_text","tweet_time","tweet_client_name","in_reply_to_userid","in_reply_to_tweetid","quoted_tweet_tweetid","is_retweet","retweet_userid","retweet_tweetid","latitude","longitude","quote_count","reply_count","like_count","retweet_count","hashtags","urls","user_mentions","poll_choices","apm"];
def dataframe_row_to_string(row, df, apm):
    result = "";
    for col in cols:
        if col in df.columns:
            result += '"' + str(row[col]).replace('"', '').replace('\n', '') + '",';
        elif col != "apm":
            result += '"",';
    return result + '"' + str(apm) + '"';

def train_model(csv):    
    # Get all file paths and their associated motives from the dataframe.
    
    print("Loading data groups");
    
    ROWS_PER_ITERATION = 250;
    x, y = [], [];
    
    for j in range(0, math.ceil(15003/ROWS_PER_ITERATION)):
        print("\tLoading data group " + str(j));
        dataframe = pd.read_csv(csv + ".csv", escapechar='\\', usecols=cols, skiprows=[i for i in range(1, 1 + (j * ROWS_PER_ITERATION))], nrows=ROWS_PER_ITERATION, converters={'tweet_text':lambda x:x.replace('\n','')});
        x += [point for point in combine_measures(dataframe)];
        y += dataframe["apm"].tolist();
    
    print("Loaded Dataset");
        
    knn = KNeighborsClassifier(n_neighbors=1);
    knn.fit(x, y);
    
    print("Fit Model");
    
    file = open(csv + "_MODEL.pickle", 'wb');
    pickle.dump(knn, file);
    file.close();
    
    print("Generated Model File");

def test_model(csv, model):    
    # Get all file paths and their associated motives from the dataframe.
              
    dataframe = pd.read_csv(csv + ".csv", escapechar='\\', usecols=cols);
    apms = dataframe["apm"].tolist();
    
    print("Loaded Dataset");
    
    x, y = [point for point in combine_measures(dataframe)], apms;
    
    print("Combined Measures");
        
    file = open(model, 'rb');
    knn = pickle.load(file);
    file.close();
    
    print("Model Loaded");
    
    predictions = knn.predict(x);
    score = metrics.accuracy_score(y, predictions);
    print("\tACCURACY SCORE: " + str(score));
    
    
# test_model("models/sampled_250_actors_model_k1_TEST_ISOLATE", "models/sampled_250_actors_model_k1.pickle");
# isolate_data_points("models/sampled_250_actors_model_k1_TEST");

train_model("models/sampled_250_actors_model_k1_TRAIN_ISOLATE");
test_model("models/sampled_250_actors_model_k1_TEST_ISOLATE", "models/sampled_250_actors_model_k1_TRAIN_ISOLATE_MODEL.pickle");
# isolate_data_points("models/sampled_250_actors_model_k1_TRAIN");