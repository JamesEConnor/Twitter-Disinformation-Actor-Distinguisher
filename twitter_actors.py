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
from train_actor_clustering import combine_measures;

# Track the run time
start_time = time.time();

# Get the handle to check
parser = ArgumentParser()
parser.add_argument('-u', '--handle',
                    help='Twitter handle to attribute to an information operation.',
                    default="thezenhaitian");
args = parser.parse_args();

if not args.handle:
    raise ValueError("Handle not specified.");

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
for tweet in tweepy.Cursor(api.user_timeline, screen_name=args.handle).items(10):
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
result["time"] = time.time() - start_time;
print(result);