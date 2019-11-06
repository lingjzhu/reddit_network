#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 22:28:12 2019

@author: lukeum
"""


import convokit
from convokit import Corpus, download
import datetime 
from collections import Counter
import json
from tqdm import tqdm
import argparse
import csv
import os

def Download_data(subreddit:str,data_dir:str,out_dir:str,sum_dir:str):
    '''
    Download a subreddit and store its monthly data
    '''
    
    # download a corpus
    corpus = convokit.Corpus(filename=download("subreddit-"+subreddit,data_dir=data_dir))
        
    # sort all utterances based on time stamp
    all_utterances = [i[1] for i in iter(corpus.utterances.items())]
    all_utterances = sorted(all_utterances,key=lambda i:i.timestamp)
    
    # extract year and month for all the time stamps
    timestamp = [datetime.datetime.fromtimestamp(i.timestamp) for i in all_utterances]
    timestamp = [(t.year,t.month) for t in timestamp]
    
    # get unique time stamps
    unique_timestamp = Counter(timestamp)
    print(len(unique_timestamp))
    unique_timestamp = sorted(unique_timestamp.items(),key=lambda i:i[0])
    
    
    # save stats
    with open(os.path.join(sum_dir,subreddit+'stats.csv'),'w') as f:
        csvwriter = csv.writer(f,delimiter=',')
        for k,v in unique_timestamp:
            csvwriter.writerow([k[0],k[1],v])
        
        
        
    # get utterances each month
    start = 0
    for k,v in tqdm(unique_timestamp):
        
        end = start+v
        subset = timestamp[start:end]
        
        # sanity check to make sure that data are from the same month
        assert len(set(subset))==1
        print('%s_%s'%(k[0],k[1]))
        
        monthly = [u.__dict__ for u in all_utterances[start:end]]
        # sanity check again to make sure that data are from the same month
        monthly_timestamp = [datetime.datetime.fromtimestamp(i['timestamp']) for i in monthly]
        monthly_timestamp = [(t.year,t.month) for t in monthly_timestamp]
        assert len(set(monthly_timestamp))==1
        
        start += v
        
        # save data into a json file for each month
        with open(os.path.join(out_dir,'%s_%s'%(k[0],k[1])),'w') as monthly_corpus:
            save_dict = []
            for u in monthly:
                if type(u['user']) == str:
                    save_dict.append(u)
                else:
                    u['user'] = u['user'].name
                    save_dict.append(u)
                    
            json.dump(save_dict,monthly_corpus,indent=0)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data","--path_to_data",dest='data_dir', help="Please specify the path to data",
                        default="./data")
    parser.add_argument("-out","--output_path",dest='out_dir', help="Please specify the output path",
                        default="./processed")
    
    
    
    args = parser.parse_args()
    
    sum_dir = './summaries'    
    if not os.path.exists(sum_dir):
        os.makedirs(sum_dir)
    
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
        
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    subreddits = ['apple','baseball','beer','boardgames','cars','FinalFantasy',
                  'Guitar','harrypotter','hockey','Liverpool','Patriots','pcgaming',
                  'photography','poker','reddevils','running','StarWars','subaru','pokemon']
    
    
    for i in subreddits:
        
        try:
            sub_out_dir = os.path.join(args.out_dir,i)
            
            if not os.path.exists(sub_out_dir):
                os.mkdir(sub_out_dir)
            Download_data(i,args.data_dir,sub_out_dir,sum_dir)
        except Exception as e:
            print(e)
            print(i)
        
        
        
        
        
        
        
    
