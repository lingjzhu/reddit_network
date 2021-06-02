#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:05:07 2019

@author: lukeum
"""


import rapidjson as json
import re
import os
import argparse
import csv
from tqdm import tqdm
import networkx as nx
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt

#path= './processed/FinalFantasy/2018_4'

def Load_monthly_data(path):
    '''
    Load the json file of the data in each month
    '''
    with open(path,'r') as json_file:
        monthly = json.load(json_file)
        
    return monthly


def Extract_users_by_thread(monthly):
    '''
    Extract all users from a particular month, grouped by threads
    '''
    threads = sorted(monthly, key=lambda i:i['root'])
    # Extract unique conversations and sort them
    roots = [i['root'] for i in threads]
    root_counter = Counter(roots)
    unique_thread = sorted(root_counter.items(),key=lambda i:i[0])
    
#    print(len(root_counter))
    
    start = 0
    all_users = {}
    for k,v in unique_thread:
        # get users in each thread
        end = start + v
        conversation = threads[start:end]
        start = end
        
        assert len(set(p['root'] for p in conversation))==1
        
        users = [p['user'] for p in conversation if p['user']!='[deleted]' and p['user']!='AutoModerator']
        all_users[k]=list(set(users))
        
    return all_users


def Create_drect_reply_graph(monthly):
    G = nx.Graph()
    
    threads = sorted(monthly, key=lambda i:i['root'])
    # Extract unique conversations and sort them
    roots = [i['root'] for i in threads]
    root_counter = Counter(roots)
    unique_thread = sorted(root_counter.items(),key=lambda i:i[0])
    
    #    print(len(root_counter))
    
    start = 0
    all_users = {}
    for k,v in unique_thread:
        # get users in each thread
        end = start + v
        conversation = threads[start:end]
        start = end
        
        assert len(set(p['root'] for p in conversation))==1
    
        users = [p['user'] for p in conversation if p['user']!='[deleted]' and p['user']!='AutoModerator']
        G.add_nodes_from(users)
        posts = {p['_id']:p for p in conversation}
    
        for post in posts.values():
        	if post['reply_to'] is not None and post['reply_to'] in posts.keys():
        		edgelist = [(post['user'],posts[post['reply_to']]['user'])]
        		G.add_edges_from(edgelist)
        
    return G



def Compute_graph_stats(all_users):
    
    G = nx.Graph()
    for con in all_users.keys():
        
        nodes = all_users[con]
        edges = combinations(nodes,2)

        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

#    density = nx.density(G)
#    num_nodes = G.number_of_nodes()
#    num_edges = G.number_of_edges()
#    avg_cc = nx.average_clustering(G)

    return G


if __name__ == "__main__":
    
 
    parser = argparse.ArgumentParser()
    parser.add_argument("-data","--path_to_data",dest='data_dir', help="Please specify the path to data",
                        default="../processed")
    parser.add_argument("-out","--output_path",dest='out_dir', help="Please specify the output path",
                        default="../data")    
    
    
    args = parser.parse_args()
        
    subred = os.listdir(args.data_dir)[:200]
    
    for sub in tqdm(subred):
#        with open(os.path.join(args.out_dir,sub+'.csv'),'w') as outfile:
#            writer = csv.writer(outfile)
#            writer.writerow(['Time','Density','Avg_CC','Nodes','Edges'])
            
        monthly_path = os.listdir(os.path.join(args.data_dir,sub))
 #       monthly_path = [m for m in monthly_path if int(re.search(r'^(\d+)_(\d+)$',m).group(1))>=2014]
        
        for m in tqdm(monthly_path):
            
            
            path = os.path.join(args.data_dir,sub,m)
            monthly = Load_monthly_data(path)
            
            all_users = Extract_users_by_thread(monthly)
            G = Compute_graph_stats(all_users)
#            G = Create_drect_reply_graph(monthly)
            
            if not os.path.exists(os.path.join(args.out_dir,sub)):
                os.mkdir(os.path.join(args.out_dir,sub))
            out_path = os.path.join(args.out_dir,sub,m)
            nx.write_adjlist(G,out_path)
            print('Saved to %s'%(out_path))
#                writer.writerow([m, density, num_nodes, num_edges])
                
    
    '''
    options = {
        'nodelist':d.keys(), 
        'node_size':[v/100 for v in d.values()],
        'node_color':'black',
        'width': 0.001,
        'alpha':0.5
    }
    nx.draw_kamada_kawai(subG,**options)
    plt.savefig("graph.png", dpi=500)
    plt.close()
    '''

    
    
    
    
    
    
    
    
    
    
    
    