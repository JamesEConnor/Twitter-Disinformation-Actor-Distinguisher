# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 12:56:21 2020

@author: James
"""
# Imports
import time;
import tweepy;
import json;
import pickle;
from argparse import ArgumentParser;
import pandas as pd;
import numpy as np;
import data.cache;

from motives.run_motive_classifier import classifyDF;
from topics.run_topic_quantifier import cluster_topics;
from methods.run_method_encoder import tweet_info;


# Combine all three measurements for KNeighborsClassifier
def combine_measures(dataframe):    
    motive_classes = classifyDF(dataframe, "motives/models/model.pickle");
    topics_quantified = cluster_topics(dataframe["tweet_text"].tolist(), "topics/models/model_001.pickle");
    methods_encoded = tweet_info(dataframe);
        
    result = [];
    for index in range(len(motive_classes)):
        append = motive_classes[index];
        append.extend([topics_quantified[index]]);
        append.extend(methods_encoded[index]);
        result.append(append);
        
    return np.asarray(result);

def get_actor(handle):
    # Track the run time
    start_time = time.time();
        
    # Check the cache for the handle.
    cache_check = data.cache.check_cache(handle);
    if cache_check.empty or cache_check.isnull().values.any():
        # Get the API credentials
        creds = {};
        with open('data/creds.json') as file:
            j = json.loads(file.read());
            creds['api_key'] = j['api_key'];
            creds['api_secret'] = j['api_secret'];
                
        
        # Construct an empty dataframe for the tweets
        df = pd.DataFrame(columns=["tweet_text", "account_creation_date", "account_language", "tweet_language", "tweet_time", "is_retweet"]);
        
        
        # Authenticate the API
        auth = tweepy.AppAuthHandler(creds['api_key'], creds['api_secret']);
        api = tweepy.API(auth);
        
        # Gather the 10 most recent tweets for the account
        for tweet in tweepy.Cursor(api.user_timeline, screen_name=handle).items(10):
            new_row = {"tweet_text": tweet.text, "account_creation_date": tweet.user.created_at, "account_language": tweet.user.lang, "tweet_language": tweet.lang, "tweet_time": tweet.created_at, "is_retweet": hasattr(tweet, 'retweeted_status')};
            df = df.append(new_row, ignore_index=True);
                
        # Load the KNN model
        MODEL = 'models/sampled_200_actors_model_k1.pickle';
        file = open(MODEL, 'rb');
        knn = pickle.load(file);
        file.close();
        
        # Quantify all of the tweets into their three measurements.
        df = combine_measures(df);
        
        # Predict for each tweet
        predictions = knn.predict(df).tolist();
        
        # Find the likelihood of the account belonging to different APMs
        percentages = {};
        for num in sorted(set(predictions), key = lambda ele: predictions.count(ele)):
            percentages[str(num)] = predictions.count(num)/len(predictions);
            
        # Construct the result and print it.
        result = {};
        result["percentages"] = percentages;
        result["run_time"] = time.time() - start_time;
        
        # Save the new data to the cache.
        data.cache.save_cache(handle, str(result));
        
        return str(result);
    else:
        # Print the cached result.
        return str(cache_check.iloc[0]);