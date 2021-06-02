#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:53:18 2020

@author: lukeum
"""


import json
import os
import re
import argparse
import spacy
import time

import pandas as pd
from tqdm import tqdm
from collections import Counter
from itertools import dropwhile
from multiprocessing import Pool



def load_monthly_data(path):
    '''
    Load the json file of the data in each month
    '''
    with open(path,'r') as json_file:
        monthly = json.load(json_file)
        
    return monthly


def get_word_and_users(p,cutoff=1):
        '''
        Generate a frequency list of users and words
        
        Parameters
        ----------
        p : TYPE
            DESCRIPTION.
        cutoff : TYPE, optional
            DESCRIPTION. The default is 2.
    
        Returns
        -------
        None.
    
        '''
        try:
            if not os.path.exists(p[2]):
                monthly = load_monthly_data(p[0])     
                word_counter = Counter()
                user_counter =  Counter()
                # count words in each month
                for post in monthly:
                    if post['text'] != '[deleted]':
                        sen = processor(post['text'])
                        # remove stop words
                        sen = [token.text.lower() for token in sen if not token.is_stop and not token.like_url and not token.is_punct]
                        word_counter.update(sen)
                    
                    if post['user']!='[deleted]' and post['user']!='AutoModerator':
                        user_counter.update([post['user']])
                    
                for k,v in dropwhile(lambda drop: drop[1] > cutoff, word_counter.most_common()):
                    del word_counter[k]
                    
                print(p[1]+'\t'+str(len(word_counter))+'\n')    
                if len(word_counter) > 500 and len(user_counter) > 50:
                    with open(p[1],'w') as f:
                        json.dump(word_counter.most_common(),f)
                    
                    with open(p[2],'w') as f:
                        json.dump(user_counter.most_common(),f)
                   
        except:
            print('Skip %s'%(p[1]))
            pass
                            
                            
def load_dict(file):
    
    lexicon = {}
    with open(file,'r') as f:
        for line in f.readlines():
#            try:
            word, count = line.strip().split('\t')
            lexicon[word] = int(count)
#            except:
#                continue
    return Counter(lexicon)



def get_paths(subreddits,args):
    
    paths = []
    for subreddit in subreddits:
        files = os.listdir(os.path.join(args.data_dir, subreddit))
        if len(files) > 0: # ignore empty folders
            # initialize folder for word counts
            wfolder = os.path.join(args.word_out_dir,subreddit)
            if not os.path.exists(wfolder): 
                os.mkdir(wfolder)
            # initialize folder for user counts    
            ufolder = os.path.join(args.user_out_dir,subreddit)
            if not os.path.exists(ufolder):
                os.mkdir(ufolder)
            for f in files:
                # generate a list of paths
                inpath = os.path.join(args.data_dir,subreddit,f)
                wpath = os.path.join(wfolder,f)
                upath = os.path.join(ufolder,f)
                paths.append((inpath,wpath,upath))
                
    return paths



def get_merge_paths(subreddits,data_dir,out_dir):
    
    paths = []
    for subreddit in subreddits:
        files = os.listdir(os.path.join(data_dir, subreddit))
        if len(files) > 0:
            outpath = os.path.join(out_dir,subreddit)
            sub_paths = []
            for f in files:
                # generate a list of paths
                inpath = os.path.join(data_dir,subreddit,f)
                sub_paths.append(inpath)
            paths.append((outpath,sub_paths))
    
    return paths


def merge_json(paths):
    
    outpath, inpath = paths
    
    out = {}
    for p in inpath:
        time = re.search(r'/(\d+_\d+)$',p).group(1)
        partition = json.load(open(p,'r'))
        out[time] = partition
    
    with open(outpath,'w') as f:
        json.dump(out,f)
        


def remove_files(paths):
    
     outpath, inpath = paths
     for p in inpath:
         os.remove(p)
         
         
def chunking(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
         
        
        
        
processor = spacy.load('en_core_web_sm',disable=["parser", "tagger","ner"])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",default='./processed',type=str)
    parser.add_argument("--word_out_dir", default='./summaries/word_counts',type=str)
    parser.add_argument("--user_out_dir", default='./summaries/user_counts',type=str)
    parser.add_argument("--word_merge_dir", default='./summaries/words',type=str)
    parser.add_argument("--user_merge_dir", default='./summaries/users',type=str)
    parser.add_argument('--chunk_size', default=2,type=int)
    args = parser.parse_args()

    
    all_subreddits = os.listdir(args.data_dir)
    processed = set(os.listdir(args.word_merge_dir))
    all_subreddits = [s for s in all_subreddits if s not in processed][::-1]
    
    
    for subreddits in chunking(all_subreddits,args.chunk_size):
        paths = get_paths(subreddits,args)
        print(len(paths))
                
#        with Pool() as P:
#            P.map(get_word_and_users,paths)
        for p in paths:
            get_word_and_users(p)
    
        wpaths = get_merge_paths(subreddits,args.word_out_dir,args.word_merge_dir)
        upaths = get_merge_paths(subreddits,args.user_out_dir,args.user_merge_dir)
        
        '''
        with Pool() as P:
            P.map(merge_json,wpaths)
            P.map(merge_json,upaths)
            P.map(remove_files,wpaths)
            P.map(remove_files,upaths)
        print('Chunk completed!')
        '''
        for wp,up in zip(wpaths,upaths):
            merge_json(wp)
            merge_json(up)
            remove_files(wp)
            remove_files(up)
        