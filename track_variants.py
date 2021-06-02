#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 08:56:25 2020

@author: lukeum
"""

import os
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import rapidjson as json
from itertools import chain

import argparse
from multiprocessing import Pool

def load_dict(file):
    
    with open(file,'r') as f:
        lexicon = json.load(f)
        
    if type(list(lexicon.values())[0]) == list:
        times = list(lexicon.keys())
        for t in times:
            lexicon[t] = {k:v for k,v in lexicon[t]}
    return lexicon


def init_empyty_df(times, slang_terms):
    
    init = np.zeros((len(times),len(slang_terms)))
    variations = pd.DataFrame(init,columns=slang_terms,index=times)
    
    return variations


def get_vocab(lexicon):
    
    vocab = chain.from_iterable([list(v.keys()) for v in lexicon.values()])
    return set(vocab)
    
def count_variants(subr):
    
    print('Processing %s'%(subr))
    try:
        lexicon = load_dict(os.path.join(path,subr))
        
        times = sorted(list(lexicon.keys()))
        vocab = get_vocab(lexicon)
        slangs = list(vocab.intersection(all_slangs))
        
        
        variations = init_empyty_df(times, slangs)
        
        
        for time in times:
            words = lexicon[time]
            
            intersect = set(words.keys()).intersection(slangs)
            
            for w in intersect:
                variations.loc[time][w] = words[w]
        
        variations.to_csv(os.path.join(outpath,subr))
    except:
        print('Skip %s'%(subr))
        pass
            


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",default='./summaries/words/',type=str)
    parser.add_argument("--out_dir", default='./summaries/variants',type=str)
    parser.add_argument("--slangs",default='./all_slangs',type=str)
    args = parser.parse_args()
    
    path = args.data_dir
    outpath = args.out_dir
    
    all_slangs = []
    with open(args.slangs,'r') as f:
        for i in f.readlines():
            all_slangs.append(i.strip())
    all_slangs = set(all_slangs)


    
    subreddits = set(os.listdir(path)).difference(set(os.listdir(outpath)))
    
    with Pool() as P:
     
        P.map(count_variants, subreddits)
     

        
        
    
    
    
    
    
    
    
    
    
    
    
    