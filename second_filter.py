#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:29:40 2020

@author: lukeum
"""

import pandas as pd
import os
from tqdm import tqdm
from collections import Counter

path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/word_count_csv'

files = [os.path.join(path,f) for f in os.listdir(path)]


words = Counter()
for f in tqdm(files):
    data = pd.read_csv(f,index_col=0)
    total = data.sum(axis=0)
    total = total[total>=10].sort_values(ascending=False)
    if len(total) > 0:
        words.update(total.to_dict())
    else:
        print('Skip %s'%f)
        
with open('./summaries/filtered_words','w') as f:
    for k,v in words.most_common():
        f.write('%s\t%s\n'%(k,v))
