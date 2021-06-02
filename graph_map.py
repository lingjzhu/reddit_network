#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:05:07 2019

@author: lukeum
"""


import json
import re
import os
import argparse
import csv
from tqdm import tqdm
import networkx as nx
from collections import Counter
from itertools import combinations
from multiprocessing import Pool

def load_monthly_data(path):
    '''
    Load the json file of the data in each month
    '''
    with open(path,'r') as json_file:
        monthly = json.load(json_file)
        
    return monthly


def extract_conversations_by_thread(monthly):
    '''
    Extract all users from a particular month, grouped by threads
    '''
    threads = sorted(monthly, key=lambda i:i['root'])
    # Extract unique conversations and sort them
    roots = [i['root'] for i in threads]
    root_counter = Counter(roots)
    unique_thread = sorted(root_counter.items(),key=lambda i:i[0])
    users = set(i['user'] for i in threads if i['user'] !='[deleted]' and i['user']!='AutoModerator')
#    print(len(root_counter))
    
    start = 0
    all_threads = {}
    for k,v in unique_thread:
        # get users in each thread
        end = start + v
        posts = threads[start:end]
        start = end
        
        assert len(set(p['root'] for p in posts))==1
        
        all_threads[k]=list(posts)
        
    return all_threads, list(users)



def get_path_from_leaf_to_root(leaf_utt, root_utt, utterances):
    """
    Helper function for get_root_to_leaf_paths, which returns the path for a given leaf_utt and root_utt
    """
    try:
        if len(root_utt) == 1:
            root_utt = root_utt[0]
            if leaf_utt == root_utt:
                return [leaf_utt]
            path = [leaf_utt]
            root_id = root_utt['_id']
            while leaf_utt['reply_to'] != root_id:
                path.append(utterances[leaf_utt['reply_to']])
                leaf_utt = path[-1]
            path.append(root_utt)
            return path[::-1]
        
        else: # if there are multiple roots
            root_id = [utt['_id'] for utt in root_utt]
            if leaf_utt['_id'] in root_id:
                return [leaf_utt]
            path = [leaf_utt]
            while leaf_utt['_id'] not in root_id:
                path.append(utterances[leaf_utt['reply_to']])
                leaf_utt = path[-1]
            return path[::-1]
    except:
        return []



def get_root_to_leaf_paths(utterances):
    """
    Get the paths (stored as a list of lists of utterances) from the root to each of the leaves
    in the conversational tree
    :return: List of lists of Utterances
    """

    utt_reply_tos = {utt['_id']: utt['reply_to'] for utt in utterances if utt['reply_to'] is not None}
    target_utt_ids = set(list(utt_reply_tos.values()))
    speaker_utt_ids = set(list(utt_reply_tos.keys()))
    root_utt_id = target_utt_ids - speaker_utt_ids # There should only be 1 root_utt_id: None
    
    if len(root_utt_id) == 1: # only one root node
        root_utt = [utt for utt in utterances if utt['reply_to'] is None or utt['_id']==str(root_utt_id)]
        if len(root_utt) != 0: # the root node should exsit in this month
            
            leaf_utt_ids = speaker_utt_ids - target_utt_ids
        
            utterances = {utt['_id']:utt for utt in utterances}
            paths = [get_path_from_leaf_to_root(utterances[leaf_utt_id], root_utt, utterances)
                     for leaf_utt_id in leaf_utt_ids]
            return paths
        else: 
            return []
        
    elif len(root_utt_id) > 1:
        root_utt_id = list(root_utt_id)
        root_utt = [utt for utt in utterances if utt['reply_to'] in root_utt_id]
        leaf_utt_ids = speaker_utt_ids - target_utt_ids
        utterances = {utt['_id']:utt for utt in utterances}
        paths = [get_path_from_leaf_to_root(utterances[leaf_utt_id], root_utt, utterances) 
                 for leaf_utt_id in leaf_utt_ids]
        return paths


def get_graph(paths,users):
    
    G = nx.Graph()
    G.add_nodes_from(users)
    for branch in paths: 
        if len(branch) > 1 and len(branch) <=4:
            nodes = [u['user'] for u in branch if u['user'] !='[deleted]' and u['user']!='AutoModerator']
            edges = combinations(nodes,2)
            G.add_edges_from(edges)            
        elif len(branch) > 4:
            for i in range(3,len(branch)):
                nodes = [u['user'] for u in branch[i-3:i] if u['user'] !='[deleted]' and u['user']!='AutoModerator']
                edges = combinations(nodes,2)
                G.add_edges_from(edges)   
            
    return G



def get_all_branches(threads):
    
    branches = []
    for k, utterances in threads.items():
        if len(utterances) > 1:
            branch = get_root_to_leaf_paths(utterances)    
            
            if branch is not None:
                branches += branch
    return branches


def generate_graphs_from_threads(path):
    try:
        in_path,out_path = path
        data = load_monthly_data(in_path)
        threads, users = extract_conversations_by_thread(data)
        branches = get_all_branches(threads)
        G = get_graph(branches, users)
        G.remove_edges_from(nx.selfloop_edges(G))
        if G.number_of_nodes() >= 50:
            nx.write_adjlist(G,out_path)
            print('Saved to %s'%(out_path))
    except:
        print('skip %s'%out_path)
        pass

if __name__ == "__main__":
    
 
    parser = argparse.ArgumentParser()
    parser.add_argument("-data","--path_to_data",dest='data_dir', help="Please specify the path to data",
                        default="./processed")
    parser.add_argument("-out","--output_path",dest='out_dir', help="Please specify the output path",
                        default="./summaries/graph")    
    
    
    args = parser.parse_args()
        
    subreddits = os.listdir(args.data_dir)
    
    subreddits = ['baseball']
    paths = []
    for subreddit in tqdm(subreddits):
        files = os.listdir(os.path.join(args.data_dir, subreddit))
        if len(files) > 0: # ignore empty folders
            # initialize folder for word counts
            gfolder = os.path.join(args.out_dir,subreddit)
            if not os.path.exists(gfolder): 
                os.mkdir(gfolder)
            for f in files:
                # generate a list of paths
                inpath = os.path.join(args.data_dir,subreddit,f)
                gpath = os.path.join(gfolder,f)
                paths.append((inpath,gpath))
                
    print(len(paths))
#    with Pool() as P:
#        P.map(generate_graphs_from_threads,paths)
    for p in tqdm(paths):
        generate_graphs_from_threads(p)
    
    