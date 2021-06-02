#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 09:38:58 2020

@author: lukeum
"""

import os
import re
import json
import networkx as nx
import numpy as np
import argparse

from collections import defaultdict, Counter
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from scipy import sparse
from itertools import chain
from tqdm import tqdm



def get_paths(data_dir):
    
    subreddits = os.listdir(data_dir)
    all_paths = {}

    for sr in subreddits:
        sub_path = os.path.join(data_dir,sr)
        all_paths[sr] = [os.path.join(sub_path,month) for month in os.listdir(sub_path)]
    
    time_stamps = set(re.search('/(\d+_\d+)$',p).group(1) for p in chain.from_iterable(list(all_paths.values())))
    
    monthly_paths = {}
    for time_stamp in time_stamps:
        paths = {subreddit:p for subreddit, path in all_paths.items() for p in path if re.search(time_stamp,p)}
        if len(paths) > 0:
            monthly_paths[time_stamp] = paths
            
    return monthly_paths
    

def load_monthly_data(path):
    '''
    Load the json file of the data in each month
    '''
    with open(path,'r') as json_file:
        monthly = json.load(json_file)
        
    if type(monthly) == list:
        monthly = {k:v for k,v in monthly}    
    return monthly



def get_active_users(sr, cutoff=3):
    
    monthly = load_monthly_data(sr[1])
    counter = Counter(monthly)
    users = [k for k,v in counter.items() if v >= cutoff]
    
    return (sr[0], users), len(counter)


def generate_graph(paths):
    
#    try:
    month, subreddits = paths
    subreddits = list(subreddits.items())
    all_users = []
    user_counts = []
    for sr in subreddits:
        d, num_users = get_active_users(sr)
        all_users.append(d)
        user_counts.append(num_users)
    user_counts = np.array(user_counts)
    X = [k for k,v in all_users]
    Y = [' '.join(v) for k,v in all_users]    
    
    tokenizer = RegexpTokenizer('\s+',gaps=True)
    vocab = list(set(i for k,v in all_users for i in v))

    vectorizer = CountVectorizer(vocabulary=vocab,lowercase=False,tokenizer=tokenizer.tokenize)    
    vectorizer.fit(Y)    
    m = vectorizer.transform(Y)    
    
    adj = m@m.T
    diagonals = adj.diagonal()
    adj = adj - sparse.diags(diagonals)
#    adj = adj / user_counts[np.newaxis,:]
    adj = adj.toarray()    
    
#    G = nx.from_numpy_matrix(adj,create_using=nx.DiGraph)
    G = nx.from_numpy_matrix(adj)
    labels = {i:k for i,k in enumerate(X)}
    G = nx.relabel_nodes(G,labels)    
    
    graph = nx.readwrite.json_graph.node_link_data(G)
    with open(os.path.join(out_dir,month),'w') as f:
        json.dump(graph,f)
    print(('%s:  $s nodes')%month,(G.number_of_nodes()))
#    except:
#        print(month+'failed!')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",dest='data_dir', help="Please specify the path to data",
                        default="./summaries/user_counts")
    parser.add_argument("--out_dir",dest='out_dir', help="Please specify the output path",
                        default="./summaries")    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    out_dir = args.out_dir
    
    
    all_paths = get_paths(data_dir)
    print(len(all_paths))
    all_paths = list(all_paths.items())
    
    with Pool() as P:
        P.map(generate_graph,all_paths)
    '''
    for paths in tqdm(all_paths):
        try:
            generate_graph(paths)
        except:
            continue
    '''    


    
    
    
    
    
    
    
    