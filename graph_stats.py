#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:29:47 2020

@author: lukeum
"""

import argparse
import os
import networkx as nx
import networkit as nk
import numpy as np
import pandas as pd
import rapidjson as json
from networkx.readwrite import json_graph
#import powerlaw
from multiprocessing import Pool
from tqdm import tqdm



def graph_measures(G):

    density = nk.graphtools.density(G)
    nodes = G.numberOfNodes()
    edges = G.numberOfEdges()
    clustering_coef = np.mean(nk.centrality.LocalClusteringCoefficient(G).run().scores())
    global_cluster = nk.globals.ClusteringCoefficient.exactGlobal(G)
    
    largest = nk.components.ConnectedComponents.extractLargestConnectedComponent(G, True)
    largest_percent = largest.numberOfNodes()/nodes

    degrees = nk.centrality.DegreeCentrality(G).run().scores()
    assortativity = nk.correlation.Assortativity(G,degrees).run().getCoefficient()
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)
    
    degrees = np.array(degrees)
    singletons = np.sum(degrees==0)/nodes
    

    return {'nodes':nodes,'edges':edges,'density':density,'assortativity':assortativity,
            'local_cluster':clustering_coef, 'global_cluster':global_cluster, 'avg_degree':avg_degree,'max_degree':max_degree,'min_degree':min_degree,
            'large_com':largest_percent,'singletons':singletons}
    
def init_empyty_df(times):
    
    attributes = ['nodes','edges','density','assortativity','local_cluster','global_cluster' ,'avg_degree',
                  'max_degree','min_degree','large_com','singletons']
    init = np.zeros((len(times),len(attributes)))
    variations = pd.DataFrame(init,columns=attributes,index=times)
    
    return variations


def analyze_graphs(path):
    
    inpath, outpath = path
    
    if not os.path.exists(outpath):
        try:
            with open(inpath,'r') as f:
                all_graphs = json.load(f)
            
            times = list(all_graphs.keys())
            data = init_empyty_df(sorted(times))
            
            for t, g in all_graphs.items():
                G = json_graph.node_link_graph(g)
            #nx.write_gexf(nxG,'2018_10.gexf')
                G = nk.nxadapter.nx2nk(G)
                G.removeSelfLoops()
            #o = nk.overview(G)
                measures = graph_measures(G)
                data.loc[t] = measures
            
            data.to_csv(outpath)
            
            print('Save to %s'% outpath)
        except:
            print('Skip %s'% s)
            pass


def analyze_inter_graphs(path):
    
    inpath, outpath = path
    
    times = os.listdir(inpath)

    data = init_empyty_df(sorted(times))
    
    for t in times:
        with open(os.path.join(inpath,t),'r') as f:
            G = json.load(f)
        G = json_graph.node_link_graph(G)
    #nx.write_gexf(nxG,'2018_10.gexf')
        G = nk.nxadapter.nx2nk(G,'weight')
        G.removeSelfLoops()
    #o = nk.overview(G)
        measures = graph_measures(G)
        data.loc[t] = measures
        print("Done %s"%t)
    data.to_csv(outpath)
    

def centrality(G):
    
    btwn = nk.centrality.Betweenness(G).run().scores()
    close = nk.centrality.Closeness(G, False, nk.centrality.ClosenessVariant.Generalized).run().scores()
    deg = nk.centrality.DegreeCentrality(G).run().scores()
    ec = nk.centrality.EigenvectorCentrality(G).run().scores()
    pr = nk.centrality.PageRank(G).run().scores()
#    katz = nk.centrality.KatzCentrality(G, 0.1,1.0,1e-08).run().scores()
    
    return {"betweenness":btwn, "closeness":close, "degree":deg, "eigen":ec,
            "pagerank":pr}
    


def extract_centrality(paths):
    
    inpath, outpath = paths
    
    with open(inpath,'r') as f:
        G = json.load(f)    

    G = json_graph.node_link_graph(G)
    idmap = list(G.nodes())
    G = nk.nxadapter.nx2nk(G,'weight')
    
    cen = centrality(G)
    data = pd.DataFrame()
    
    data["Subreddits"] = idmap
    for k,v in cen.items():
        data[k] = v
        
    data.to_csv(outpath)
    print("Save to %s"%outpath)
        
        




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",default='./summaries/graphs',type=str)
    parser.add_argument("--out_dir", default='./summaries/graph_stats',type=str)
    parser.add_argument("--graph_type",default="individual",type=str)
    args = parser.parse_args()
    
    if args.graph_type == "individual":
        files = os.listdir(args.data_dir)
        paths = []
        for f in files:
            inpath = os.path.join(args.data_dir,f)
            outpath = os.path.join(args.out_dir,f)
            paths.append((inpath,outpath))
    
        with Pool() as P:
            P.map(analyze_graphs,paths)
        
    else:  
        inpath = args.data_dir
        outpath = args.out_dir
        
#        inpath = "./summaries/bgraph3"
#        outpath = "./summaries/bgraph3_stats"
#        analyze_inter_graphs((inpath,outpath+'/total'))
        
        times = os.listdir(inpath)
        paths = [(os.path.join(inpath,t),os.path.join(outpath,t)) for t in times]        
        for p in tqdm(paths):
            extract_centrality(p)
