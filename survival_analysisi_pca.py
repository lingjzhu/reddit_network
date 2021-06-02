#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 08:44:55 2020

"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import argparse
from tqdm import tqdm
# For preprocessing
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler, PowerTransformer
from sklearn_pandas import DataFrameMapper 
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.models import LogisticHazard,MTLR,DeepHitSingle
from pycox.evaluation import EvalSurv

from lifelines import CoxPHFitter

def load_data(path):
    df_all = pd.read_csv(path)
    df_all = df_all.dropna()
    subreddits = df_all['subreddit'].unique().tolist()
    subreddits = shuffle(subreddits,random_state=666)
    train = subreddits[:3500]
    val = subreddits[3500:3865]
    test = subreddits[3865:]
    df_test = df_all[df_all['subreddit'].isin(test)]
    df_train = df_all[df_all['subreddit'].isin(train)]
    df_val = df_all[df_all['subreddit'].isin(val)]
    return df_train,df_test,df_val




def transform_data(df_train,df_test,df_val, mod, scale, cols_standardize, log_columns, num_durations=100):
    
    tf_train = df_train.copy()
    tf_test = df_test.copy()
    tf_val = df_val.copy()
    if scale == "minmax":
        standardize = [([col], MinMaxScaler()) for col in cols_standardize]
    elif scale == "standard":
        standardize = [([col], StandardScaler()) for col in cols_standardize]
    elif scale == "robust":
        standardize = [([col], RobustScaler()) for col in cols_standardize]
    elif scale == "power":
        standardize = [([col], PowerTransformer()) for col in cols_standardize]

    if len(log_columns) != 0:
        log_scaler = lambda x: np.log(np.abs(x)+1e-7)
        
        for c in log_columns:
            tf_train.loc[:,c] = log_scaler(tf_train.loc[:,c])
            tf_val.loc[:,c] = log_scaler(tf_val.loc[:,c])
            tf_test.loc[:,c] = log_scaler(tf_test.loc[:,c])
   
    x_mapper = DataFrameMapper(standardize)
    
    x_train = x_mapper.fit_transform(tf_train).astype('float32')
    x_val = x_mapper.transform(tf_val).astype('float32')
    x_test = x_mapper.transform(tf_test).astype('float32')
    
    pca = PCA(n_components=10,whiten=True)
    x_train = pca.fit_transform(x_train)
    x_val = pca.transform(x_val)
    x_test = pca.transform(x_test)
    
    if mod == "LogisticHazard":
        labtrans = LogisticHazard.label_transform(num_durations)
    elif mod == "MTLR":
        labtrans = MTLR.label_transform(num_durations)
    elif mod == "DeepHitSingle":
        labtrans = DeepHitSingle.label_transform(num_durations) 
    
    get_target = lambda tf: (tf['duration'].values.astype("float32"), tf['event'].values)
    y_train = labtrans.fit_transform(*get_target(tf_train))
    y_val = labtrans.transform(*get_target(tf_val))
    
    train = (x_train, y_train)
    val = (x_val, y_val)
    
    # We don't need to transform the test labels
    durations_test, events_test = get_target(tf_test)
    
    return x_mapper, labtrans, train, val, x_test, durations_test, events_test, pca



def initialize_model(dim,labtrans,in_features):
    num_nodes = [dim,dim]
    out_features = labtrans.out_features
    batch_norm = True
    dropout = 0.1
    
    net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)


    #model = MTLR(net, tt.optim.Adam, duration_index=labtrans.cuts)
    model = LogisticHazard(net, tt.optim.Adam, duration_index=labtrans.cuts)
    #model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)

    return model

def plot_survival(pc_train,pc_col,model,outpath,duration=100,baseline=False):
    
    for i in list(pc_col):
#        avg = pd.DataFrame(pc_train[pc_col].median(axis=0)).transpose()
        avg = pd.DataFrame(pc_train[pc_col].mean(axis=0)).transpose()
        avg = pd.concat([avg]*duration,ignore_index=True)
        for j in range(duration):
            avg.loc[j,i] = pc_train[i].quantile([(j+1)/duration]).values
            
#        avg_value = x_mapper.transform(avg).astype("float32")
        avg_value = avg.values
        if baseline == False:
            surv = model.predict_surv_df(torch.tensor(avg_value).float())
        elif baseline == True:
            avg.loc[:,pc_col] = avg_value
            surv = model.predict_survival_function(avg)
        #surv.iloc[:, :].plot(drawstyle='steps-post')
        ax = plt.imshow(surv.T,origin='lower')
        plt.colorbar(ax, extend='both')
        plt.ylabel('Percentile')
        plt.xlabel('Relative life span')
#        plt.legend()
        plt.title(i)
        plt.savefig(outpath+"/%s.png"%(i),dpi=300)
        plt.close()
        surv.to_csv(outpath+"/"+"%s"%(i))


    


cols_standardize = ['nodes','edges','density','avg_degree','max_degree','large_com','singletons','betweenness','closeness',
                 'degree','eigen','pagerank','adjusted_assort','adjusted_gc','adjusted_lc']



'''

cols_standardize = ['nodes','edges','density','assortativity','global_cluster','local_cluster',
                 'avg_degree','max_degree','large_com','adjusted_assort','adjusted_gc','adjusted_lc']

cols_standardize = ['betweenness','closeness',
                 'degree','eigen','pagerank']

log_columns =  ['nodes','edges', 'density','avg_degree','max_degree','degree','closeness', 'pagerank','betweenness','eigen']


cols_standardize = ['assortativity','global_cluster','local_cluster',
                 'large_com','singletons','adjusted_assort','adjusted_gc','adjusted_lc']


log_columns =  ['nodes','edges', 'density','avg_degree','max_degree']

log_columns =  ['degree','closeness', 'pagerank','betweenness','eigen']
'''

#cols_standardize = ['nodes','density','betweenness','closeness','degree','eigen','pagerank']

log_columns =  ['nodes','edges', 'density','avg_degree','max_degree','degree','closeness', 'pagerank','betweenness','eigen']

#log_columns = []

names = ['Nodes','Edges','Density','Avg.degree','Max.degree','Large.com','Singletons','Betweenness','Closeness',
                 'Degree','Eigenvector','Pagerank','Adj.assort','Adj.gc','Adj.lc']
weights = pca.components_  


for i in range(1,11):

    plt.figure(figsize=(10,8))    
    plt.rcParams.update({'font.size': 26})
    ax = plt.barh(names,weights[i-1],tick_label=names)
    [i.set_color("salmon") for n,i in enumerate(ax) if n in [7,8,9,10,11]]
    plt.title('PC%s'%(i))
    plt.tight_layout()
    plt.savefig('./survival/pca/surpc%s.pdf'%(i),dpi=300)
    plt.close()
    

#plt.figure(figsize=(20,10))    
plt.rcParams.update({'font.size': 30})
fig, ax = plt.subplots(nrows=1,ncols=5,sharey=True,figsize=(30,9))
ax[0].barh(names,weights[0],tick_label=names)
ax[0].set_title('PC1')
for i in [7,8,9,10,11]:
    ax[0].get_children()[i].set_color('salmon') 
#[i.set_color("salmon") for n,i in enumerate(ax[0]) if n in [7,8,9,10,11]]
ax[1].barh(names,weights[1],tick_label=names)
ax[1].set_title('PC2')
for i in [7,8,9,10,11]:
    ax[1].get_children()[i].set_color('salmon') 
ax[2].barh(names,weights[2],tick_label=names)
ax[2].set_title('PC3')
for i in [7,8,9,10,11]:
    ax[2].get_children()[i].set_color('salmon') 
ax[3].barh(names,weights[5],tick_label=names)
ax[3].set_title('PC6')
for i in [7,8,9,10,11]:
    ax[3].get_children()[i].set_color('salmon') 
ax[4].barh(names,weights[6],tick_label=names)
ax[4].set_title('PC7')
for i in [7,8,9,10,11]:
    ax[4].get_children()[i].set_color('salmon') 
plt.tight_layout()
plt.savefig('./figures/sur_pcs.pdf',dpi=300)

#plt.figure(figsize=(20,10))    
plt.rcParams.update({'font.size': 30})
fig, ax = plt.subplots(nrows=1,ncols=5,sharey=True,figsize=(30,9))
ax[0].barh(names,weights[3],tick_label=names)
ax[0].set_title('PC4')
for i in [7,8,9,10,11]:
    ax[0].get_children()[i].set_color('salmon') 
#[i.set_color("salmon") for n,i in enumerate(ax[0]) if n in [7,8,9,10,11]]
ax[1].barh(names,weights[4],tick_label=names)
ax[1].set_title('PC5')
for i in [7,8,9,10,11]:
    ax[1].get_children()[i].set_color('salmon') 
ax[2].barh(names,weights[7],tick_label=names)
ax[2].set_title('PC8')
for i in [7,8,9,10,11]:
    ax[2].get_children()[i].set_color('salmon') 
ax[3].barh(names,weights[8],tick_label=names)
ax[3].set_title('PC9')
for i in [7,8,9,10,11]:
    ax[3].get_children()[i].set_color('salmon') 
ax[4].barh(names,weights[9],tick_label=names)
ax[4].set_title('PC10')
for i in [7,8,9,10,11]:
    ax[4].get_children()[i].set_color('salmon') 
plt.tight_layout()
plt.savefig('./figures/sur_pcs_more.pdf',dpi=300)



# some hyperparameters
batch_size = 2048
epochs = 3
seeds = [666]#,233,6666,2333,66666,23333,88,888,8888,168]
scalers = ["standard"]
#models = ["LogisticHazard","MTLR","DeepHitSingle"]
models = ["LogisticHazard"]
hiddens = [256]
lrs = [0.001]

# initiate an empty dataframe
results = pd.DataFrame(columns=["random","model","hiddens","lr","scalers","c-index","brier","nll"])

df_train,df_test,df_val = load_data("./summaries/survival_data")

for seed in seeds:
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    for scale in scalers:
        for mod in models:
            x_mapper, labtrans, train, val, x_test, durations_test, events_test, pca = transform_data(
                df_train,df_test,df_val, mod, scale, cols_standardize, log_columns, num_durations=100)
            x_train, y_train = train
            for dim in hiddens:
                for lr in lrs:
                    outpath = "./survival/%s_%s_%s_%s_%s"%(mod,scale,dim,lr,seed)
                    if not os.path.exists(outpath):
                        os.mkdir(outpath)
                    
                    in_features = x_train.shape[1]
                    model = initialize_model(dim,labtrans,in_features)
                    model.optimizer.set_lr(0.001)
    
                    callbacks = [tt.callbacks.EarlyStopping()]
                    log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)
                    
                    surv = model.predict_surv_df(x_test)
                    ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
                    
                    result = pd.DataFrame([[0]*8],columns=["random","model","hiddens",
                                                   "lr","scalers","c-index","brier","nll"])
                    
                    result["c-index"] = ev.concordance_td('antolini')  
                    print(ev.concordance_td('antolini') )
                    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
                    result["brier"] = ev.integrated_brier_score(time_grid) 
                    print(ev.integrated_brier_score(time_grid) )
                    result["nll"] = ev.integrated_nbll(time_grid) 
                    result["lr"] = lr
                    result["model"] = mod
                    result["scaler"] = scale
                    result["random"] = seed
                    result["hiddens"] = dim
                    
                    results = pd.concat([results,result],ignore_index=True)
                    results.to_csv(os.path.join(outpath,"results"))
                
                    pc_col = ['PC'+str(i) for i in range(x_train.shape[1])]
                    pc_train = pd.DataFrame(x_train,columns = pc_col)
                    plot_survival(pc_train,pc_col,model,outpath,duration=100)



'''
Baseline Cox model

'''
def run_baseline(runs=10):
    concordance = []
    ibs = []
    
    for i in tqdm(range(runs)):
        df_train,df_test,df_val = load_data("./summaries/survival_data")
        
        x_mapper, labtrans, train, val, x_test, durations_test, events_test, pca = transform_data(
            df_train,df_test,df_val,'LogisticHazard', "standard", cols_standardize, log_columns, num_durations=100)
        x_train, y_train = train

        
        cols = ['PC'+str(i) for i in range(x_train.shape[1])] + ['duration','event']
        pc_col = ['PC'+str(i) for i in range(x_train.shape[1])]
        cox_train = pd.DataFrame(x_train,columns = pc_col)
        cox_test = pd.DataFrame(x_test,columns=pc_col)
        
#        cox_train.loc[:,pc_col] = x_train
        cox_train.loc[:,["duration"]] = y_train[0]
        cox_train.loc[:,'event'] = y_train[1]
#        cox_train = cox_train.drop(columns=[i for i in list(df_train) if i not in cols])
#        cox_test.loc[:,pc_col] = x_test
#        cox_test = cox_test.drop(columns=[i for i in list(df_train) if i not in cols])
        cox_train = cox_train.dropna()
        cox_test = cox_test.dropna()
        cph = CoxPHFitter().fit(cox_train, 'duration', 'event')
#        cph.print_summary()
        surv = cph.predict_survival_function(cox_test)
        ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')
        concordance.append(ev.concordance_td('antolini')) 
        time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
        ibs.append(ev.integrated_brier_score(time_grid))
        
        print("Average concordance: %s"%np.mean(concordance))
        print("Average IBS: %s"%np.mean(ibs))
    
    plot_survival(cox_train,
                  pc_col,cph,'./survival/cox',baseline=True)
'''
Plotting

lol: 131
ifttt: 36
'''




def plot_word_survival()
    data = pd.DataFrame()
    for i in [1731244,1731736]:
        word = pd.DataFrame(df_test.loc[i,:]).transpose()
        data = pd.concat([data,word])
    
    if len(log_columns) != 0:
        log_scaler = lambda x: np.log(np.abs(x)+1e-7)
     
    for c in log_columns:
         data.loc[:,c] = log_scaler(data.loc[:,c].values.astype(np.float32))
    feats = x_mapper.transform(data).astype('float32')
    feats = pca.transform(feats)
    
    surv = model.predict_surv_df(feats)
    time = surv.index
    plt.rcParams.update({'font.size': 12.5})
    plt.figure(figsize=(7,3.5))
#    sns.set_style('dark')
    plt.plot(surv.iloc[:,0],linewidth=2,label='lol - 131 months (LH)')
    plt.plot(surv.iloc[:,1],linewidth=2,label='ifttt - 36 months (LH)')
    
    cox_feats = data.copy()
    cox_feats.loc[:,pc_col] = feats
    cox_feats.drop(columns=['word','subreddit','life-span','min_degree'])
    surv = cph.predict_survival_function(cox_feats)
    surv.index = time[2:]
    plt.plot(surv.iloc[:,0],'--',linewidth=2,label='lol - 131 months (Cox)')
    plt.plot(surv.iloc[:,1],'--',linewidth=2,label='ifttt - 36 months (Cox)')
    plt.legend()
    plt.ylabel("$S(t|x)$")
    plt.xlabel("Time (months)")
    plt.tight_layout()
    plt.savefig('./figures/word_survival.pdf',dpi=300)
    plt.close()
    
    
    
def plot_heatmap():
    nodes = pd.read_csv('./survival/LogisticHazard_standard_256_0.001_666/PC2',index_col=0)
    avg_degree = pd.read_csv('./survival/LogisticHazard_standard_256_0.001_666/PC4',index_col=0)
    ic_degree = pd.read_csv('./survival/LogisticHazard_standard_256_0.001_666/PC7',index_col=0)
    density = pd.read_csv('./survival/LogisticHazard_standard_256_0.001_666/PC9',index_col=0)
    
    plt.rcParams.update({'font.size': 15})
    nodes.index = labtrans.cuts
    avg_degree.index = labtrans.cuts
    density.index = labtrans.cuts
    ic_degree.index = labtrans.cuts
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(8,4),sharex=True, sharey=True,
                        gridspec_kw={'hspace': 0.3, 'wspace':0.1},squeeze=True)
    im = ax1.imshow(nodes.transpose(),origin='lower',extent=[0,152,0,100], aspect='auto')
    ax1.set_title('PC3')
    ax1.margins(0.05)
    ax2.imshow(avg_degree.transpose(),origin='lower',extent=[0,152,0,100], aspect='auto')
    ax2.set_title('PC5')
    ax3.imshow(ic_degree.transpose(),origin='lower',extent=[0,152,0,100], aspect='auto')
    ax3.set_title('PC8')
    ax4.imshow(density.transpose(),origin='lower',extent=[0,152,0,100], aspect='auto')
    ax4.set_title('PC10')
    for ax in fig.get_axes():
        ax.label_outer()
    plt.xlim(0,130)
    cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax,extend='both',label=r"$S(t|x)$")
    fig.text(0.03, 0.5, r"Percentile (%): low $\rightarrow$ high", rotation="vertical", va="center")
    fig.text(0.42, 0.02, "Time (months)", va="center")
#    fig.tight_layout()
    plt.savefig("./figures/more_variables.pdf",dpi=300)
    plt.close()
    

    
def plot_heatmap():
    var1 = pd.read_csv('./survival/LogisticHazard_standard_256_0.001_66666/large_com',index_col=0)
    var2 = pd.read_csv('./survival/LogisticHazard_standard_256_0.001_66666/singletons',index_col=0)
    var3 = pd.read_csv('./survival/LogisticHazard_standard_256_0.001_66666/local_cluster',index_col=0)
    var4 = pd.read_csv('./survival/LogisticHazard_standard_256_0.001_66666/global_cluster',index_col=0)
    
    plt.rcParams.update({'font.size': 15})
    var1.index = labtrans.cuts
    var2.index = labtrans.cuts
    var3.index = labtrans.cuts
    var4.index = labtrans.cuts
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(8,4),sharex=True, sharey=True,
                        gridspec_kw={'hspace': 0.3, 'wspace':0.1},squeeze=True)
    im = ax1.imshow(var1.transpose(),origin='lower',extent=[0,152,0,100], aspect='auto')
    ax1.set_title('Largest Commponent')
    ax1.margins(0.05)
    ax2.imshow(var2.transpose(),origin='lower',extent=[0,152,0,100], aspect='auto')
    ax2.set_title('Singletons')
    ax3.imshow(var3.transpose(),origin='lower',extent=[0,152,0,100], aspect='auto')
    ax3.set_title('Clust.Coef.')
    ax4.imshow(var4.transpose(),origin='lower',extent=[0,152,0,100], aspect='auto')
    ax4.set_title('Transitivity')
    for ax in fig.get_axes():
        ax.label_outer()

    cb_ax = fig.add_axes([0.93, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax,extend='both',label=r"$S(t|x)$")
    fig.text(0.03, 0.5, r"Percentile (%): low $\rightarrow$ high", rotation="vertical", va="center")
    fig.text(0.42, 0.02, "Time (months)", va="center")
#    fig.tight_layout()
    plt.savefig("./figures/variables.pdf",dpi=300)
    plt.close()
    
    
    
    
    
    
    
    
    
    
    
    