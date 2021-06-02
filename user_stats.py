#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:21:33 2020

@author: lukeum
"""

import rapidjson as json
import pandas as pd
import os
from tqdm import tqdm



def load_dict(file):
    
    with open(file,'r') as f:
        lexicon = json.load(f)
        
    if type(list(lexicon.values())[0]) == list:
        times = list(lexicon.keys())
        for t in times:
            lexicon[t] = {k:v for k,v in lexicon[t]}
    return lexicon


path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/users'

subreddits = os.listdir(path)

columns = ['time','subreddits','users','posts']

data = pd.DataFrame(columns=columns)

for s in tqdm(subreddits):
    
    
    users = load_dict(os.path.join(path,s))
    
    for month, counts in users.items():
        
        placeholder = pd.DataFrame([[month, s, len(counts), sum(counts.values())]],columns=columns)
        
        data = data.append(placeholder,ignore_index=True)
        
data.to_csv('./summaries/user_stats')        
        