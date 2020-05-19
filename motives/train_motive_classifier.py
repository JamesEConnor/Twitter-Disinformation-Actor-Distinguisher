# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:31:51 2020

@author: James

@summary: Trains a Multinomial Naive Bayes Classifier for classifying the motives behind a disinformation tweet.
"""

import sys;

import re;
import random;

from textblob import TextBlob;
import pandas as pd;

import pickle;
import nltk;
from nltk.corpus import stopwords;
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB

URL_REGEX = re.compile(
    # protocol identifier
    u"(?:(?:https?|ftp)://)"
    # user:pass authentication
    u"(?:\S+(?::\S*)?@)?"
    u"(?:"
    # IP address exclusion
    # private & local networks
    u"(?!(?:10|127)(?:\.\d{1,3}){3})"
    u"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
    u"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
    # IP address dotted notation octets
    # excludes loopback network 0.0.0.0
    # excludes reserved space >= 224.0.0.0
    # excludes network & broadcast addresses
    # (first & last IP address of each class)
    u"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
    u"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
    u"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
    u"|"
    # host name
    u"(?:(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)"
    # domain name
    u"(?:\.(?:[a-z\u00a1-\uffff0-9]-?)*[a-z\u00a1-\uffff0-9]+)*"
    # TLD identifier
    u"(?:\.(?:[a-z\u00a1-\uffff]{2,}))"
    u")"
    # port number
    u"(?::\d{2,5})?"
    # resource path
    u"(?:/\S*)?"
    , re.UNICODE);

# Removes hashtags, urls, etc from tweets using regex.
def clean_tweet(tweet):
    try:
        # Remove URLs
        tweet = URL_REGEX.sub("", tweet);
    except:
        print("BAD 1: " + str(tweet));
        return None;
            
    # Remove hashtags
    tweet = ' '.join([split for split in tweet.split(' ') if not split.startswith('#')])
    
    return tweet;

def document_features(document, w_features):
    document_words = set(document);
    features = {};
    for word in w_features:
        features['contains(%s)' % str(word)] = (word in document_words);
        
    return features;

# Trains a classifier for recognizing the motive of a disinformation tweet.
def train_motives_classifier(files, classes, saveModel, trainSplit=100, maxTrainPerFile=50):
    # Initialize variables
    corpus = [];
    all_words = [];
    stp = stopwords.words("english");
    
    # Create the set of data by looping through each file and associating each tweet with 
    for index in range(len(files)):
        print("FILE " + str(index) + ": " + files[index]);
        csv = pd.read_csv(files[index], usecols=["tweet_text"], nrows=maxTrainPerFile);
        
        # Loop through each tweet, remove punctuation/links/mentions/hashtags
        # Translate each tweet to english to ensure consistency
        # Tokenize each tweet for NLP
        # Add to the corpus
        for line in csv["tweet_text"]:
            #Attempt to translate, accounting for exceptions that arise as a result of not needing to translate.
            try:
                line = str(TextBlob(line).translate());
            except:
                pass;
            
            try:
                line = clean_tweet(line);
                #line = re.sub('[^\w\s]', '', line); #Remove punctuation
                line = re.sub("\d+", " ", line);  #Remove digits

                if not line:
                    continue;
            except:
                print("BAD 3: " + str(line));
                continue;
            
            try:
                line = [i.lower() for i in list(set(nltk.word_tokenize(line)) - set(stp))];
            except:
                print("BAD 3: " + str(line));
                continue;
                
            all_words += line;
            corpus.append((line, classes[index]));

    print("CORPUS: " + str(len(corpus)));

    counter = 0;
    for t, c in corpus:
        if c == 'ideological':
            counter = counter + 1;

    print(counter);

    # Shuffle the corpus to ensure adequate training and a realistic test set
    random.shuffle(corpus);
    
    # Get the featuresets mapping doc features to classifications
    word_features = list(all_words);    
    featuresets = [(document_features(doc, word_features), classification) for (doc, classification) in corpus];
    print(len(featuresets));
        
    # Split into a training and a test set.
    train_set, test_set = featuresets[:trainSplit], featuresets[trainSplit:];
    
    # Create and train the classifier, printing the accuracy.
    classifier = SklearnClassifier(MultinomialNB());
    classifier.train(train_set);
    print("Classifier accuracy:", (nltk.classify.accuracy(classifier, test_set))*100)
    
    # Save the classifier to the desired location.
    save_classifier = open(saveModel, 'wb');
    pickle.dump(classifier, save_classifier);
    save_classifier.close();
    
    # Save the word features to the same location as the classifier.
    save_word_features = open(saveModel.replace(".pickle", "_wf.pickle"), 'wb');
    pickle.dump(word_features, save_word_features);
    save_word_features.close();
    
    
    
# Read the CSV file mapping all tweet data to a motive.
labeledDataPath = "../data/actors_and_motives.csv";
df = pd.read_csv(labeledDataPath, usecols=["tweet_docs", "motive"], converters={"tweet_docs": lambda x: x.strip("[]").split(", ")});

# Removes leading and ending quote characters
df["tweet_docs"] = [[x.strip('\"') for x in df["tweet_docs"][i]] for i in range(len(df["tweet_docs"]))];

# Get all file paths and their associated motives from the dataframe.
_files, _classes = [], [];
for i in range(len(df["tweet_docs"])):
    for j in range(len(df["tweet_docs"][i])):
        _files.append(df["tweet_docs"][i][j]);
        _classes.append(df["motive"][i]);
        
# Train a model
train_motives_classifier(files=_files, classes=_classes, saveModel="models/model_001.pickle", trainSplit=50, maxTrainPerFile=20);