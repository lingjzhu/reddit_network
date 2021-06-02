#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 08:44:55 2020

@author: lukeum
"""

import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm
import ray

ray.init()


'''
Merge between-community graphs
'''
def merge_graph_data(inpath,outpath):
    bgraph = pd.DataFrame()
    files = os.listdir(inpath)
    
    for t in files:
        
        data = pd.read_csv(os.path.join(inpath,t),index_col=0)
        data["time"] = [t for i in range(len(data))] 
    
        bgraph = pd.concat([bgraph,data],ignore_index=True)
    
    bgraph.to_csv(outpath+"bgraphs_stats_2",index=False)


'''
Generate word survival data
'''
columns = ["word","subreddit","life-span","duration","event",'nodes', 'edges', 'density', 
           'assortativity', 'local_cluster','global_cluster', 'avg_degree', 'max_degree', 
           'min_degree', 'large_com','singletons', 'betweenness', 'closeness', 'degree', 
           'eigen', 'pagerank', 'r_assort','r_gc','r_lc','adjusted_assort',
           'adjusted_gc','adjusted_lc'
           ]

@ray.remote
def generate_survival_data(subr):

    # get word statistics
    word_counts = pd.read_csv(os.path.join(word_path,subr),index_col=0)
    selected_words = set(list(word_counts))
    selected_words = selected_words.intersection(filtered_words)
    word_counts = word_counts.loc[:,list(selected_words)]
    # get graph statistics
    graph = pd.read_csv(os.path.join(graph_path,subr),index_col=0)
    baseline = pd.read_csv(os.path.join(baseline_path,subr),index_col=0)
    baseline.rename(columns={'assortativity':'r_assort', 
                             'local_cluster':'r_lc',
                             'global_cluster':'r_gc'},
                              inplace=True)
    graph['r_assort'] = [0 for i in range(len(graph))]
    graph['r_lc'] = [0 for i in range(len(graph))]
    graph['r_gc'] = [0 for i in range(len(graph))]
    graph.loc[baseline.index,list(baseline)] = baseline.values
    graph['adjusted_assort'] = (graph['assortativity']-graph['r_assort'])/graph['r_assort']
    graph['adjusted_gc'] = (graph['global_cluster']-graph['r_gc'])/graph['r_gc']
    graph['adjusted_lc'] = (graph['local_cluster']-graph['r_lc'])/graph['r_lc']

    # initialize an empty dataframe
    survival_whole = pd.DataFrame(columns=columns)
    # exclude short-lived subreddits
    if len(word_counts) < 6:
        pass
    else:
        survival = pd.DataFrame([[0 for i in range(len(columns))]],columns=columns)
        # get all words
        words = list(word_counts)
        
        for word in words:
            word_stats = word_counts[word_counts[word]!=0]
            # exclude short-lived words
            if len(word_stats) <= 3:
                continue
            else:
                # extract relevant statistics
                graph_data = graph.loc[word_stats.index,:]
                graph_data = graph_data.mean(axis=0)
                survival.loc[:,graph_data.index] = graph_data.values
                survival.loc[:,"life-span"] = len(word_counts)
                survival.loc[:,"word"] = word
                survival.loc[:,"subreddit"] = subr
                survival.loc[:,"duration"] = len(word_stats)
                
                if word_stats.index[-1] < word_counts.index[-6]:
                    survival.loc[:,"event"] = 1
                else:
                    survival.loc[:,"event"] = 0
                
                bgraph_data = bgraph[bgraph["Subreddits"]==subr]
                bgraph_data = bgraph_data[bgraph_data["time"].isin(word_stats.index)]
                bgraph_data = bgraph_data.loc[:,['betweenness', 'closeness', 'degree', 'eigen', 'pagerank']].mean(axis=0)
                survival.loc[:,bgraph_data.index] = bgraph_data.values
                               
                survival_whole = pd.concat([survival_whole,survival],ignore_index=True)
        survival_whole.to_csv(os.path.join(outpath,subr))
        print("Done %s"%subr)



'''
Merge survival data
'''

def merge_survival_data(inpath,outpath):
    survival = pd.DataFrame()
    files = os.listdir(inpath)
    
    for t in tqdm(files):
        
        data = pd.read_csv(os.path.join(inpath,t),index_col=0)
        data = data.drop(columns=['r_gc','r_lc','r_assort'])
        survival = pd.concat([survival,data],ignore_index=True)
    
    survival = survival[survival['adjusted_gc']!=np.inf]
    survival.to_csv(outpath+"survival_data",index=False)


if __name__ == "__main__":
    
    filtered_words = pd.read_csv("./summaries/filtered_words",sep='\t',header=None)
    filtered_words = set(filtered_words[0].unique().tolist())
    
    word_path = './summaries/word_count_csv'
    graph_path = "./summaries/graph_stats"
    baseline_path = './summaries/graph_baseline'
    outpath = "./summaries/survival_stats"
    subreddits = os.listdir(word_path)
    
    merge_graph_data('./summaries/intercom-2-stats',"./summaries/")
    bgraph= pd.read_csv("./summaries/bgraphs_stats_2")

#    with Pool() as P:
#        P.map(generate_survival_data,subreddits)
    result_ids = []
    for paths in subreddits:
        result_ids.append(generate_survival_data.remote(paths))         

    print(ray.get(result_ids))        
    merge_survival_data(outpath,'./summaries/')
