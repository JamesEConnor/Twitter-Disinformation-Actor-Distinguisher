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

def test_model():    
    # Get all file paths and their associated motives from the dataframe.
              
    dataframe = pd.read_csv("models/sampled_250_actors_model_k1_TEST.csv");
    apms = [-1 if math.isnan(x) else x for x in dataframe["apm"].tolist()];
    
    print("Loaded Dataset");
    
    x, y = [point for point in combine_measures(dataframe)], apms;
    
    print("Combined Measures");
        
    file = open("models/sampled_250_actors_model_k1.pickle", 'rb');
    knn = pickle.load(file);
    file.close();
    
    print("Model Loaded");
    
    predictions = knn.predict(x);
    score = metrics.accuracy_score(y, predictions);
    print("\tACCURACY SCORE: " + str(score));
    
    
test_model();