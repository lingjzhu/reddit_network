#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:10:49 2020

@author: lukeum
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from multiprocessing import Pool



columns = ["Time","Subreddits", "Words", 'nodes', 'edges', 'density', 
           'assortativity', 'local_cluster','global_cluster', 'avg_degree', 'max_degree', 
           'min_degree', 'large_com','singletons', 'betweenness', 'closeness', 'degree', 
           'eigen', 'pagerank','users','posts','r_assort','r_gc','r_lc']




word_path = './summaries/word_count_csv'
graph_path = "./summaries/graph_stats"
baseline_path = './summaries/graph_baseline'
outpath = "./summaries/innovation_stats"
subreddits = os.listdir(word_path)

bgraph= pd.read_csv("./summaries/bgraphs_stats_2")
activities = pd.read_csv('./summaries/user_stats',index_col=1)
activities = activities.drop(columns=['Unnamed: 0'])
activities.rename(columns={'subreddits':'Subreddits'},
                              inplace=True)


subreddits = os.listdir(word_path)

filtered_words = pd.read_csv("./summaries/filtered_words",sep='\t',header=None)
filtered_words = set(filtered_words[0].unique().tolist())

def generate_innovation_stats(s):
    

    # get word statistics
    word_counts = pd.read_csv(os.path.join(word_path,s),index_col=0)
    selected_words = set(list(word_counts))
    selected_words = selected_words.intersection(filtered_words)
    word_counts = word_counts.loc[:,list(selected_words)]
    # get graph statistics
    graph = pd.read_csv(os.path.join(graph_path,s),index_col=0)
    # get baseline statistics
    baseline = pd.read_csv(os.path.join(baseline_path,s),index_col=0)
    baseline.rename(columns={'assortativity':'r_assort', 
                             'local_cluster':'r_lc',
                             'global_cluster':'r_gc'},
                              inplace=True)
    
    data = pd.DataFrame(np.zeros((len(graph),len(columns))),columns=columns)
    
    data["Time"] = graph.index
    data = data.set_index("Time")
    data.loc[graph.index,list(graph)] = graph.values
    
    inter = bgraph[bgraph["Subreddits"]==s]
    inter = inter.set_index("time")
    data.loc[inter.index,list(inter)] = inter.values
    
    users = activities[activities['Subreddits']==s]
    data.loc[users.index,list(users)] = users.values
    
    data.loc[baseline.index,list(baseline)] = baseline.values
    
    
    
    
    
    for w in list(word_counts):
        series = word_counts.loc[:,w]
        series = series[series!=0]
        initial = sorted(series.index)[0]
        data.loc[initial,"Words"] += 1
    
    data.to_csv(os.path.join(outpath,s))
    print("%s done!"%s)
    

with Pool() as P:
    P.map(generate_innovation_stats,subreddits)


def merge_innovation_data(inpath,outpath):
    innovation = pd.DataFrame()
    files = os.listdir(inpath)
    
    for t in tqdm(files):
        
        data = pd.read_csv(os.path.join(inpath,t))
        innovation = pd.concat([innovation,data],ignore_index=True)
    innovation= innovation[innovation["Subreddits"]!="0.0"]

    innovation['adjusted_assort'] = (innovation['assortativity']-innovation['r_assort'])/innovation['r_assort']
    innovation['adjusted_gc'] = (innovation['global_cluster']-innovation['r_gc'])/innovation['r_gc']
    innovation['adjusted_lc'] = (innovation['local_cluster']-innovation['r_lc'])/innovation['r_lc']
    innovation['activity'] = np.log(innovation['posts']/innovation['users'])
    innovation = innovation.dropna()
    innovation = innovation[innovation["adjusted_lc"]!=np.inf]
    innovation.to_csv(outpath+"innovation_data",index=False)


merge_innovation_data("./summaries/innovation_stats","./summaries/")

innovation = pd.read_csv("./summaries/innovation_data")

#sns.distplot(innovation["adjusted_gc"])
#plt.xscale("log")
#plt.yscale("log")

'''
Add delta features
'''


selected = ['nodes', 'edges', 'density', 
            'avg_degree', 'max_degree', 'large_com', 'singletons', 'betweenness', 
            'closeness', 'degree', 'eigen', 'pagerank', 
            'adjusted_assort', 'adjusted_gc', 'adjusted_lc']

d_selected = ['d_'+i for i in selected]


def merge_innovation_data(inpath,outpath):
    innovation = pd.DataFrame()
    files = os.listdir(inpath)
    
    for t in tqdm(files):
        
        data = pd.read_csv(os.path.join(inpath,t))
        data['adjusted_assort'] = (data['assortativity']-data['r_assort'])/data['r_assort']
        data['adjusted_gc'] = (data['global_cluster']-data['r_gc'])/data['r_gc']
        data['adjusted_lc'] = (data['local_cluster']-data['r_lc'])/(data['r_lc']+1e-4)
        data['activity'] = np.log(data['posts']/data['users'])
        data = data.sort_values('Time')
        
        feat = data.loc[:,selected]
        last = np.zeros_like(feat)
        last[1:,:] = feat.to_numpy()[:-1,:]
        delta = feat.to_numpy() - last
        data.loc[:,d_selected] = delta
        
        innovation = pd.concat([innovation,data],ignore_index=True)
    innovation= innovation[innovation["Subreddits"]!="0.0"]
    
    innovation = innovation.replace([np.inf,-np.inf],np.nan)
    innovation = innovation.dropna()
#    innovation = innovation[innovation["adjusted_lc"]!=np.inf]
    innovation.to_csv(outpath+"innovation_data_delta",index=False)




merge_innovation_data("./summaries/innovation_stats","./summaries/")










