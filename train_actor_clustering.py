# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:19:52 2020

@author: James
"""

import pandas as pd;
import numpy as np;
import pickle;
import random;

from motives.run_motive_classifier import classifyDF;
from topics.run_topic_quantifier import cluster_topics;
from methods.run_method_encoder import tweet_info;

from sklearn.neighbors import KNeighborsClassifier;
from sklearn import metrics;

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

def train_model():
    # Get measures of all tweets from dataframe.    
    # Read the CSV file mapping all tweet data to a motive.
    labeledDataPath = "data/actors_and_motives.csv";
    df = pd.read_csv(labeledDataPath, usecols=["tweet_docs", "apm"], converters={"tweet_docs": lambda x: x.strip("[]").split(", ")});
    
    # Removes leading and ending quote characters
    df["tweet_docs"] = [[x.strip('\"') for x in df["tweet_docs"][i]] for i in range(len(df["tweet_docs"]))];
    
    # Create the training and testing datasets
    TRAIN_SPLIT = 250;
    MAX_LINES = 300;
    RANDOM_SELECTION = 200;
    dataset = [];
    
    # Get all file paths and their associated motives from the dataframe.
    print("GATHERING DATASET...")
    counter = 0;
    for i in range(len(df["tweet_docs"])):
        for j in range(len(df["tweet_docs"][i])):        
            read_dataframe = pd.read_csv(df["tweet_docs"][i][j], nrows=MAX_LINES);
            
            if RANDOM_SELECTION < len(read_dataframe):
                read_dataframe = read_dataframe.sample(n=RANDOM_SELECTION);
            
            dataset += [(measure, df["apm"][i]) for measure in combine_measures(read_dataframe)];
            
            counter += 1;
            print("Loading Dataset " + str(counter));
    
    print("SHUFFLING DATASET...")
    random.shuffle(dataset);
    x_training, y_training = np.asarray([point[0] for point in dataset[TRAIN_SPLIT:]]), np.asarray([point[1] for point in dataset[TRAIN_SPLIT:]]);
    x_testing, y_testing = np.asarray([point[0] for point in dataset[:TRAIN_SPLIT]]), np.asarray([point[1] for point in dataset[:TRAIN_SPLIT]]);
    
    # Create, train, and test for the best classifier.
    k_range = range(1, 25);
    best_model = None;
    best_score = 0;
    chosen_k = 1;
    
    # Loop through several values of k and find the best performing model.
    for k in k_range:
        print("TRAINING ON K-VALUE: " + str(k));
        knn = KNeighborsClassifier(n_neighbors=k);
        knn.fit(x_training, y_training);
        prediction_testing = knn.predict(x_testing);
        score = metrics.accuracy_score(y_testing, prediction_testing);
        print("\tACCURACY SCORE: " + str(score));
        
        if score > best_score:
            best_model = knn;
            best_score = score;
            chosen_k = k;
    
    print("\n\nBEST MODEL SCORE: " + str(best_score));
    print("BEST K-VALUE: " + str(chosen_k));
    
    # Save the classifier.
    save_file = open('models/sampled_' + str(RANDOM_SELECTION) + '_actors_model_k' + str(chosen_k) + '.pickle', 'wb');
    pickle.dump(best_model, save_file);
    save_file.close();
    
    print("MODEL SAVED");
    
    
# train_model();