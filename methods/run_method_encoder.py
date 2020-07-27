# -*- coding: utf-8 -*-
"""
Created on Fri May  1 09:45:52 2020

@author: James
"""
import pandas as pd;
from datetime import datetime;
import math;

#Example column
#5.33622E+17	299148448	Maria Luis	marialuis91	Nantes, France	journaliste indépendante/un vrai journaliste est un chômeur		8012	1450	5/15/2011	en	fr	@bellisarobz Ces photos illustrent parfaitement ce que ressentent les pères à un concert de One Direction http://t.co/YGdg8ihIh7	11/15/2014 14:07	Twitter Web Client	5.33622E+17	574356455		FALSE					0	0	0	0		[http://fr.awdnews.com/divertissements/5757-ces-photos-illustrent-parfaitement-ce-que-ressentent-les-p%C3%A8res-%C3%A0-un-concert-de-one-direction.html]	[574356455]	
def account_info(path, userid):    
    # Read data from CSV
    use_cols = ["userid", "user_profile_description", "account_creation_date", "account_language", "tweet_language", "tweet_text", "tweet_time", "tweet_client_name", "latitude", "longitude", "hashtags", "urls", "user_mentions", "is_retweet"];
    csv_data = pd.read_csv(path, usecols=use_cols);
    users_and_tweets = dict(iter(csv_data.groupby('userid')))
    
    # Summarize account info
    result = [];    
    result += [strDateToUTC(users_and_tweets[userid]["account_creation_date"]).iloc[0]];
    result += [averageLanguageCode(users_and_tweets[userid]["account_language"])];
    result += [averageLanguageCode(users_and_tweets[userid]["tweet_language"])];
    result += [averageTime(users_and_tweets[userid]["tweet_time"])];
    result += [retweetRatio(users_and_tweets[userid]["is_retweet"])];
    
    return result;
        
def tweet_info(df):
    # Summarize tweet info
    result = [];
    for index, tweet in df.iterrows():
        try:
            append = [];
            append += [strDateToUTC(tweet["account_creation_date"])];       # The account creation date in UTC days
            append += [averageLanguageCode(tweet["account_language"])];             # The account language, tokenized
            append += [averageLanguageCode(tweet["tweet_language"])];               # The tweet language, tokenized
            append += [encodeTime(tweet["tweet_time"])];                           # The tweet time, formatted nicely
            append += [1 if tweet["is_retweet"] == True else 0];            # Whether it's a retweet (1 for True, 0 for False)
            result.append(append);
        except:
            print(tweet["tweetid"]);
            pass;
    
    return result;
        
def strDateToUTC(date):
    return (pd.to_datetime(date) - datetime(1970, 1, 1)).days;
    
def averageTime(df): 
    #Converts string row to datetime row
    df = pd.to_datetime(df);
    
    #Prints mean of datetime row
    return encodeTime(df.mean());

def encodeTime(time):
    dec, integer = math.modf(float(pd.to_datetime(time).strftime('%H.%M')));
    dec /= 0.60;
    return integer + dec;

#The 34 different twitter language codes.
languages = ["en", "ar", "bn", "cs", "da", "de", "el", "es", "fa", "fi", "fil", "fr", "he", "hi", "hu", "id", "it", "ja", "ko", "msa", "nl", "no", "pl", "pt", "ro", "ru", "sv", "th", "tr", "uk", "ur", "vi", "zh-cn", "zh-tw"];
def averageLanguageCode(df):
    try:
        result = 0;
        for languageCode in df:
            if languageCode not in languages:
                continue;
            
            result += languages.index(languageCode);
            
        return result/len(df);
    except:
        return 0;

def retweetRatio(df):
    return len(df[df == True])/len(df);

# df = pd.read_csv("D:\\APC CSVs\\twitter\\iran_201906_3_tweets_csv_hashed.csv", nrows=2);
# print(tweet_info(df));