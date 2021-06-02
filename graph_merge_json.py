#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 19:27:04 2020

@author: lukeum
"""

import os
import json
import re
import networkx as nx
from multiprocessing import Pool

def merge_graphs_into_json(p):
    
    outpath, inpath = p
    
    graphs= {}
    
    for i in inpath:
        G = nx.read_adjlist(i)
        jfile = nx.node_link_data(G)
        
        time = re.search(r'/(\d+_\d+)$',i).group(1)
        graphs[time] = jfile
    
    with open(outpath,'w') as out:
        json.dump(graphs,out)
        

if __name__ == "__main__":
    

    data_dir = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/thread_graph'    
    outdir = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/TG'
        
        
    subreddits = os.listdir(data_dir)
    
    
    paths = []
    for subreddit in subreddits:
        files = os.listdir(os.path.join(data_dir, subreddit))
        outpath = os.path.join(outdir,subreddit)
        if not os.path.exists(outpath):
            sub_paths = []
            for f in files:
                # generate a list of paths
                inpath = os.path.join(data_dir,subreddit,f)
                sub_paths.append(inpath)
            
            paths.append((outpath,sub_paths))               
    print(len(paths))
    
    
    with Pool() as P:
        P.map(merge_graphs_into_json,paths)
        
        
#   Test
#    graphs = json.load(open(outpath,'r'))
#    G = nx.node_link_graph(graphs['2016_01'])
