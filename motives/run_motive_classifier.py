# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:28:02 2020

@author: James
"""

import time;

import pickle;
import pandas as pd;
from textblob import TextBlob;

# Get the document features of a piece of text.
def document_features(document, word_features):
	document_words = set(document)
	features = {}
	for word in word_features:
		features['contains(%s)' % word] = (word in document_words)
	return features

# Get the document features of a dataframe row.
def document_features_row(row, word_features):
    return document_features(row['translated'], word_features)

# Translate the text of a dataframe row.
def translate_row(row):
    try:
        return TextBlob(row["tweet_text"]).translate().text;
    except:
        return row["tweet_text"];

# Classify a single tweet
def classify(dataPoint, model):
    try:
        dataPoint = TextBlob(dataPoint).translate();
    except:
        pass;
    
    file_classifier = open(model, 'rb');
    classifier = pickle.load(file_classifier);
    file_classifier.close();
    
    file_wf = open(model.replace(".pickle", "_wf.pickle"), 'rb');
    word_features = pickle.load(file_wf);
    file_wf.close();
    
    return classifier.prob_classify(document_features(dataPoint, word_features));

# The labels of possible motive classifications
labels = ["political", "ideological"];

# Classify an entire pandas DataFrame.
def classifyDF(df, model):
    file_classifier = open(model, 'rb');
    classifier = pickle.load(file_classifier);
    file_classifier.close();
        
    file_wf = open(model.replace(".pickle", "_wf.pickle"), 'rb');
    word_features = pickle.load(file_wf);
    file_wf.close();
    
    df['translated'] = df.apply(lambda row: translate_row(row), axis=1);
    # print(df['translated']);
    df['features'] = df.apply(lambda row: document_features_row(row, word_features), axis=1);
    
    return [[x.prob(y) for y in labels] for x in classifier.prob_classify_many(df['features'])];

# Get the number of data points belonging to each classification within a file.
def classifyFile(file_data, model, maxlines=None):
    file_classifier = open(model, 'rb');
    classifier = pickle.load(file_classifier);
    file_classifier.close();
        
    file_wf = open(model.replace(".pickle", "_wf.pickle"), 'rb');
    word_features = pickle.load(file_wf);
    file_wf.close();
    
    classifications = {"error": 0};
    
    data = pd.read_csv(file_data, usecols=["tweet_text"], nrows=maxlines);
    
    for tweet in data["tweet_text"]:
        try:
            tweet = TextBlob(tweet).translate();
        except:
            pass;
        
        try:
            motive = classifier.classify(document_features(tweet, word_features));
            if motive in classifications:
                classifications[motive] = classifications[motive] + 1;
            else:
                classifications[motive] = 1;
        except:
            classifications["error"] = classifications["error"] + 1;
            pass;
            
    return classifications;
    
    
    
    
    
### TESTING SINGLE CLASSIFICATION
# probs = classify("Commended for no longer saying 'China virus' Did US military bring #Covid19 to 7th Military World Games Oct18-27, 2019 Wuhan, China? Patient zero: Maatja Benassi US Athlete/Intelligence Officer? Did World's military take it back to their countries?", "models/model.pickle");
# #probs = classify("BREAKING | Boris Johnson will get lung ventilation - health source sptnkne.ws/BWtv #SputnikBreaking @BorisJohnson", "models/model.pickle");
# print(probs.prob("political"));
# print(probs.prob("ideological"));
    

### TESTING FILE CLASSIFICATIONS
# # Read the CSV file mapping all tweet data to a motive.
# labeledDataPath = "../data/actors_and_motives.csv";
# df = pd.read_csv(labeledDataPath, usecols=["tweet_docs", "motive"], converters={"tweet_docs": lambda x: x.strip("[]").split(", ")});

# # Removes leading and ending quote characters
# df["tweet_docs"] = [[x.strip('\"') for x in df["tweet_docs"][i]] for i in range(len(df["tweet_docs"]))];

# # Get all file paths and their associated motives from the dataframe.
# _files, _classes = [], [];
# for i in range(len(df["tweet_docs"])):
#     for j in range(len(df["tweet_docs"][i])):
#         _files.append(df["tweet_docs"][i][j]);
#         _classes.append(df["motive"][i]);

# totalCorrect, total = 0, 0;
# for index in range(len(_files)):
#     classifications = classifyFile(_files[index], "models/model.pickle", maxlines=500);
#     totalCorrect += (classifications[_classes[index]] if _classes[index] in classifications else 0);
#     total += sum(classifications.values());
    
#     print(_files[index]);
#     print("CORRECT (" + _classes[index] + "): " + str((classifications[_classes[index]] if _classes[index] in classifications else 0)/sum(classifications.values())));
#     print("TOTAL ACCURACY: " + str(totalCorrect/total) + "\n");
    
# df = pd.read_csv("D:\\APC CSVs\\twitter\\iran_201906_3_tweets_csv_hashed.csv", nrows=2, usecols=["tweet_text"]);
# print(classifyDF(df, "models/model.pickle"));