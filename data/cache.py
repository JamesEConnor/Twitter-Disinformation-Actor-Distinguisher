# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 17:11:05 2020

@author: James
"""

import time;
import pandas as pd;

MAX_TIME = 120; #hours

# Open cache file and purge any outdated entries.
file = open("data/cache.csv", 'r');
df = pd.read_csv(file);
file.close();

# Function for checking the cache for data associated with a specific handle.
def check_cache(handle):
    return df['data'].where(df['handle'] == handle);
    
# Function for saving new data for a handle.
def save_cache(handle, data):
    global df;
    if handle in df['handle']:
        df.at[handle, 'data'] = data;
        df.at[handle, 'time'] = time.time();
    else:
        df = df.append({"handle": handle, "data": data, "time": time.time()}, ignore_index=True);
    
    df.to_csv("data/cache.csv", index=False);
    
# Function for purging any value that hasn't been assigned beyond a specific date.
def purge_cache(time_in_hours):
    global df;
    for index, row in df.iterrows():
        if time.time() - row['time'] > time_in_hours * 3600:
            df = df[df.handle != row['handle']];
            
purge_cache(MAX_TIME);