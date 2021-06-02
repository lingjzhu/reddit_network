#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 22:05:07 2019

@author: lukeum
"""

import os
import rapidjson as json
from tqdm import tqdm
from collections import defaultdict

data_dir = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/users'
subreddits = os.listdir(data_dir)

path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/users_split'

# split jsons

for sub in tqdm(subreddits):
    
    out_folder = os.path.join(path,sub)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    user_dict = json.load(open(os.path.join(data_dir,sub),'r'))
    for month, users in user_dict.items():
        out = os.path.join(out_folder,month)
        with open(out,'w') as f:
            json.dump(users,f)
'''
# merge jsons
for sub in tqdm(subreddits):
    
    out_file = os.path.join(path,sub)
    user_dict = defaultdict()
    times = os.listdir(os.path.join(data_dir,sub))
    for t in times:
        monthly = os.path.join(data_dir,sub,t)
        with open(monthly,'r') as f:
            data = json.load(f)
        user_dict[t] = data
        
    with open(out_file,'w') as f:
        json.dump(user_dict,f)
		
'''