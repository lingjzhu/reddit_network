#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:53:18 2020

@author: lukeum
"""


import rapidjson as json
import os
import re
import argparse
import spacy
import time

import pandas as pd
from tqdm import tqdm
from collections import Counter,defaultdict
from multiprocessing import Pool
from spacy.lang.en import English


def load_monthly_data(path):
    '''
    Load the json file of the data in each month
    '''
    with open(path,'r') as json_file:
        monthly = json.load(json_file)
        
    return monthly


def get_word_and_users(p,cutoff=1):

#    try:
#   if not os.path.exists(p[1]):
    monthly = load_monthly_data(p[0])
    monthly = sorted(monthly, key=lambda i:i['user'])
    users = [i['user'] for i in monthly]
    users = Counter(users)
    unique_users = sorted(users.items(),key=lambda i:i[0])
    dissemination =  defaultdict()
    # count words in each month
    start = 0
    
    slangs = set(pd.read_csv(p[2]))
    for k,v in unique_users:
        end = start + v
        user_posts = monthly[start:end]
        start = end
        assert len(set(p['user'] for p in user_posts))==1
        counter = Counter()
        if k!='[deleted]' and k!='AutoModerator':
            for post in user_posts:
                sen = processor(post['text'])
                # remove stop words
                sen = [token.text.lower() for token in sen if not token.is_stop and not token.like_url and not token.is_punct]
                counter.update(sen)
            
            intersect = set(counter.keys()).intersection(slangs)
            if len(intersect) != 0:
                dissemination[k] = {w:counter[w] for w in intersect}
                
        else:
            continue
            
    
    with open(p[1],'w') as f:
        json.dump(dissemination,f)
    print('save to %s'%(p[1]))
#    except:
#        print('skip %s'%(p[1]))
#        pass
                                        
def get_paths(subreddits,args):
    
    paths = []
    for subreddit in tqdm(subreddits):
        files = os.listdir(os.path.join(args.reference_dir, subreddit))
        if len(files) > 0: # ignore empty folders
            # initialize folder for word counts
            wpath = os.path.join(args.dict_dir,subreddit)
            outfolder = os.path.join(args.out_dir,subreddit)
            if not os.path.exists(outfolder): 
                os.mkdir(outfolder)
            for f in files:
                # generate a list of paths
                inpath = os.path.join(args.data_dir,subreddit,f)
                outpath = os.path.join(outfolder,f)
#                if not os.path.exists(outpath):
                paths.append((inpath,outpath,wpath))
                
    return paths




        
        
        
processor = spacy.load('en_core_web_sm',disable=["parser", "tagger","ner"])
#nlp = English()
#processor = nlp.Defaults.create_tokenizer(nlp)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",default='./processed',type=str)
    parser.add_argument("--out_dir", default='./summaries/dissemination',type=str)
    parser.add_argument('--reference_dir',default=None,type=str)
    parser.add_argument('--dict_dir',default=None,type=str)
    args = parser.parse_args()

    
    all_subreddits = os.listdir(args.reference_dir)
    
    
#    for subreddits in chunking(all_subreddits,args.chunk_size):
#    paths = get_paths(all_subreddits,args)
#    print(len(paths))
#    with open('./summaries/all_files','w') as f:
#        for p in paths:
#            f.write('%s\t%s\t%s\n'%(p[0],p[1],p[2]))
    paths = []
    with open('./summaries/all_files','r') as f:
        for line in f.readlines():
            p1,p2,p3 = line.strip().split('\t')
            if not os.path.exists(p2):
                paths.append((p1,p2,p3))
    print(len(paths))
    with Pool() as P:
        P.map(get_word_and_users,paths)
#    for p in tqdm(paths):
#        get_word_and_users(p)


    

