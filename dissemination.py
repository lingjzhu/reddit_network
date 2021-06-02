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
import numpy as np
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


        if not os.path.exists(p[1]):
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
                    user_posts = [p['text'] for p in user_posts]
                    sen = processor(' '.join(user_posts))
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
                paths.append((inpath,outpath,wpath))
                
    return paths

def init_empyty_df(times, slang_terms):
    
    init = np.zeros((len(times),len(slang_terms)))
    variations = pd.DataFrame(init,columns=slang_terms,index=times)
    
    return variations



        
        
        
processor = spacy.load('en_core_web_sm',disable=["parser", "tagger","ner"])
#nlp = English()
#processor = nlp.Defaults.create_tokenizer(nlp)


def reduce(paths):
    try:
        inpath,repath, outpath = paths
        lexicon = list(set(pd.read_csv(repath,index_col=0)))
        
        months = os.listdir(inpath)
        
        data = init_empyty_df(months,lexicon)
        
        for m in months:
            users = load_monthly_data(os.path.join(inpath,m))
            for user, counts in users.items():
                words = list(counts.keys())
                data.loc[m,words] =  data.loc[m,words] + 1
                
        data.to_csv(outpath)
        print('Save to %s'%outpath)
    except:
        print("Skip %s"%outpath)
        pass
            
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",default='./summaries/dissemination',type=str)
    parser.add_argument("--out_dir", default='./summaries/words',type=str)
    parser.add_argument('--reference_dir',default=None,type=str)
    parser.add_argument('--dict_dir',default=None,type=str)
    parser.add_argument('--reduce',default=True,type=bool)
    args = parser.parse_args()

    
    all_subreddits = os.listdir(args.reference_dir)
    
    if not args.reduce:
#    for subreddits in chunking(all_subreddits,args.chunk_size):
        paths = get_paths(all_subreddits,args)
        print(len(paths))
                
        with Pool() as P:
            list(P.imap(get_word_and_users,paths))
    
    else:
         paths = [(os.path.join(args.data_dir,r),os.path.join(args.reference_dir,r),os.path.join(args.out_dir,r))
                  for r in all_subreddits]
         print(len(paths))
         with Pool() as P:
             P.map(reduce,paths)
#         for p in paths:
#            reduce(p)


    

