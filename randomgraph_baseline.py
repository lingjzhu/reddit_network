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
from pathos.multiprocessing import ProcessPool as Pool




def graph_measures(G,num=5):

    degrees = nk.centrality.DegreeCentrality(G).run().scores()
    
    generator = nk.generators.EdgeSwitchingMarkovChainGenerator(degrees)
    
    clustering_coef = []
    global_cluster = []
    assortativity = []
    for i in range(num):
        randomG = generator.generate()
        clustering_coef.append(np.mean(nk.centrality.LocalClusteringCoefficient(randomG).run().scores()))
        global_cluster.append(nk.globals.ClusteringCoefficient.exactGlobal(randomG))
        assortativity.append(nk.correlation.Assortativity(randomG,degrees).run().getCoefficient())

    

    return {'assortativity':np.mean(assortativity),
            'local_cluster':np.mean(clustering_coef), 'global_cluster':np.mean(global_cluster)}


    
def init_empyty_df(times):
    
    attributes = ['assortativity','local_cluster','global_cluster']
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

        




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",default='./summaries/graphs',type=str)
    parser.add_argument("--out_dir", default='./summaries/baselines',type=str)
    args = parser.parse_args()
    

    files = os.listdir(args.data_dir)
    paths = []
    for f in files:
        inpath = os.path.join(args.data_dir,f)
        outpath = os.path.join(args.out_dir,f)
        paths.append((inpath,outpath))


    with Pool() as P:
        P.map(analyze_graphs,paths)
    

