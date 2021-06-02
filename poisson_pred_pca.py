#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter,defaultdict
import seaborn as sns
from tqdm import tqdm

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PoissonRegressor
from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_poisson_deviance

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor



innovation = pd.read_csv("./summaries/innovation_data")
innovation = innovation.dropna()
innovation = innovation[innovation['Words']<1000]

features = ['Words',
#            'activity',
             'nodes',
             'edges',
             'density',
             'avg_degree',
             'max_degree',
             'large_com',
             'singletons',
             'adjusted_assort',
             'adjusted_gc',
             'adjusted_lc',
             'betweenness',
             'closeness',
             'degree',
             'eigen',
             'pagerank']

names = ['Words',
#            'activity',
             'Nodes',
             'Edges',
             'Density',
             'Avg.degree',
             'Max.degree',
             'Largest.com',
             'Singletons',
             'Adj.assortativity',
             'Adj.gc',
             'Adj.lc',
             'Betweenness',
             'Closeness',
             'Degree',
             'Eigenvector',
             'Pagerank']


def plot_corr_matrix(innovation):
    feat = innovation.loc[:,features]
    corr = feat.corr(method='spearman')
    
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        yticklabels=names
    )
    
    ax.set_xticklabels(
        ax.get_yticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    [i.set_color("red") for n,i in enumerate(ax.get_xticklabels()) if n >10]
    [i.set_color("red") for n,i in enumerate(ax.get_yticklabels()) if n >10]
    plt.tight_layout()
    
    plt.savefig('./figures/corr.pdf',dpi=300)

corr.to_csv('corr')

def scoring(estimator, df_test):
    """Score an estimator on the test set."""
    y_pred = estimator.predict(df_test)

    mse = mean_squared_error(df_test["Words"], y_pred)
    mae = mean_absolute_error(df_test["Words"], y_pred)
    # Ignore non-positive predictions, as they are invalid for
    # the Poisson deviance.
    mask = y_pred > 0
    if (~mask).any():
        n_masked, n_samples = (~mask).sum(), mask.shape[0]
    mpd = mean_poisson_deviance(df_test["Words"][mask],
                          y_pred[mask])
    return mse, mae, mpd


# def processors
log_scaler = lambda x: np.log(x+1e-6)

log_scale_transformer = make_pipeline(
    FunctionTransformer(log_scaler, validate=False),
    StandardScaler()
)

all_preprocessor = ColumnTransformer(
    [
        ("log_features",log_scale_transformer,
         ['nodes','edges', 'density','avg_degree','max_degree','degree','closeness', 
          'pagerank','betweenness','eigen']),
        ("features", StandardScaler(),
           ['adjusted_assort','adjusted_lc','adjusted_gc','large_com','singletons'])
    ],
    remainder="drop",
)




preprocessor = all_preprocessor


# PCA 
df_train, df_test = train_test_split(innovation, test_size=0.1, random_state=1)
reduce = Pipeline([("preprocessor", preprocessor)]).fit(df_train)
df_train_norm = reduce.transform(df_train)

pca = PCA(n_components=10,whiten=True).fit(df_train_norm)
train_pca= pca.transform(df_train_norm)
names = ['Nodes', 'Edges', 'Density', 'Avg degree',
         'Max degree', 'Degree', 'Closeness',
          'Pagerank', 'Betweenness', 'Eigenvector'] + ['Adj.assort', 'Adj.lc',
                                  'Adj.gc', 'Large com.', 'Singletons']

                                                 
weights = poisson_glm['PCA'].components_   

plt.figure(figsize=(6,10))    
plt.rcParams.update({'font.size': 26})
ax = plt.barh(names,weights[1],tick_label=names)
[i.set_color("salmon") for n,i in enumerate(ax) if n in [5,6,7,8,9]]

   
#plt.figure(figsize=(20,10))    
plt.rcParams.update({'font.size': 30})
fig, ax = plt.subplots(nrows=1,ncols=5,sharey=True,figsize=(30,9))
ax[0].barh(names,weights[0],tick_label=names)
ax[0].set_title('PC1')
for i in [5,6,7,8,9]:
    ax[0].get_children()[i].set_color('salmon') 
#[i.set_color("salmon") for n,i in enumerate(ax[0]) if n in [7,8,9,10,11]]
ax[1].barh(names,weights[1],tick_label=names)
ax[1].set_title('PC2')
for i in [5,6,7,8,9]:
    ax[1].get_children()[i].set_color('salmon') 
ax[2].barh(names,weights[2],tick_label=names)
ax[2].set_title('PC3')
for i in [5,6,7,8,9]:
    ax[2].get_children()[i].set_color('salmon') 
ax[3].barh(names,weights[3],tick_label=names)
ax[3].set_title('PC4')
for i in [5,6,7,8,9]:
    ax[3].get_children()[i].set_color('salmon') 
ax[4].barh(names,weights[4],tick_label=names)
ax[4].set_title('PC5')
for i in [5,6,7,8,9]:
    ax[4].get_children()[i].set_color('salmon') 
plt.tight_layout()
plt.savefig('./figures/inno_pcs.pdf',dpi=300)



results = defaultdict(list)

for i in tqdm(range(20)):
    
    df_train, df_test = train_test_split(innovation, test_size=0.1)
    
    dummy = Pipeline([
        ("preprocessor", preprocessor),
        ("PCA",PCA(n_components=4,whiten=True)),
        ("regressor", DummyRegressor(strategy='mean')),
    ]).fit(df_train, df_train["Words"])   
    
    mse, mae, mpd = scoring(dummy, df_test)
    results['mse'].append(mse)
    results['mae'].append(mae)
    results['mpd'].append(mpd)
    
print("Constant mean frequency evaluation:")
print("Avg MSE: %s"%np.mean(results['mse']))
print("Avg MAE: %s"%np.mean(results['mae']))
print("Avg MPD: %s"%np.mean(results['mpd']))


results = defaultdict(list)

for i in tqdm(range(20)):
    
    df_train, df_test = train_test_split(innovation, test_size=0.1,random_state=i+666)
    n_samples = df_train.shape[0]
    poisson_glm = Pipeline([
        ("preprocessor", preprocessor),
        ("PCA",PCA(n_components=5,whiten=True)),
        ("regressor", PoissonRegressor(alpha=1e-3, max_iter=300))
    ])
    poisson_glm.fit(df_train, df_train["Words"])
    

    mse, mae, mpd = scoring(poisson_glm, df_test)       
    results['mse'].append(mse)
    results['mae'].append(mae)
    results['mpd'].append(mpd)
    
print("PoissonRegressor evaluation:")
print("Avg MSE: %s"%np.mean(results['mse']))
print("Avg MAE: %s"%np.mean(results['mae']))
print("Avg MPD: %s"%np.mean(results['mpd']))          



names = ['Nodes', 'Edges', 'Density', 'Avg degree',
         'Max degree', 'Degree', 'Closeness',
          'Pagerank', 'Betweenness', 'Eigenvector'] + ['Adj.assort', 'Adj.lc',
                                  'Adj.gc', 'Large com.', 'Singletons']

                                                 
weights = poisson_glm['PCA'].components_   

plt.figure(figsize=(10,8))    
plt.rcParams.update({'font.size': 26})
ax = plt.barh(names,weights[4],tick_label=names)
[i.set_color("salmon") for n,i in enumerate(ax) if n in [5,6,7,8,9]]
plt.title('PC5')
plt.tight_layout()
plt.savefig('./figures/pc5.pdf',dpi=300)
plt.close()

plt.figure(figsize=(10,8))    
plt.rcParams.update({'font.size': 26})
ax = plt.barh(names,weights[1],tick_label=names)
[i.set_color("salmon") for n,i in enumerate(ax) if n in [5,6,7,8,9]]
plt.title('PC2')
plt.tight_layout()
plt.savefig('./figures/pc2.pdf',dpi=300)
plt.close()

plt.figure(figsize=(10,8))    
plt.rcParams.update({'font.size': 26})
ax = plt.barh(names,weights[2],tick_label=names)
[i.set_color("salmon") for n,i in enumerate(ax) if n in [5,6,7,8,9]]
plt.title('PC3')
plt.tight_layout()
plt.savefig('./figures/pc3.pdf',dpi=300)
plt.close()



results = defaultdict(list)

for i in tqdm(range(20)):
    
    df_train, df_test = train_test_split(innovation, test_size=0.1,random_state=i+666)

    poisson_gbrt = Pipeline([
        ("preprocessor", preprocessor),
        ("PCA",PCA(n_components=10,whiten=True)),
        ("regressor", HistGradientBoostingRegressor(loss="poisson",
                                                    max_leaf_nodes=256)),
    ])
    poisson_gbrt.fit(df_train, df_train["Words"])
    
    
    mse, mae, mpd = scoring(poisson_gbrt, df_test)        
    results['mse'].append(mse)
    results['mae'].append(mae)
    results['mpd'].append(mpd)
    
print("HGBT evaluation:")
print("Avg MSE: %s"%np.mean(results['mse']))
print("Avg MAE: %s"%np.mean(results['mae']))
print("Avg MPD: %s"%np.mean(results['mpd']))       





plt.rcParams.update({'font.size': 13})
#sns.set_style('dark')
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 2.5), sharey=True)
fig.subplots_adjust(bottom=0.2)
n_bins = 50
label, df = "test", df_test
df["Words"].hist(bins=np.linspace(0, 300, n_bins),ax=axes[0])

axes[0].set_title("Data")
axes[0].set_yscale('log')
axes[0].set_xlabel("Observed Innovations")
axes[0].set_ylim([5e0, 5e4])
axes[0].set_ylabel( "Test samples")

for idx, model in enumerate([poisson_glm, poisson_gbrt]):
    y_pred = model.predict(df)
    
    pd.Series(y_pred).hist(bins=np.linspace(0, 300, n_bins),
                           ax=axes[idx+1])
    axes[idx + 1].set(
            title=model[-1].__class__.__name__,
           yscale='log',
           xlabel="Predicted Innovations"
           )
plt.tight_layout()
plt.savefig("./figures/pred_innovations.pdf",dpi=300)    












'''
data = innovation.copy()
log_feat = ['nodes','density','avg_degree','max_degree','degree','betweenness','eigen','pagerank','closeness','large_com','singletons']
feat =  ['assortativity','local_cluster','global_cluster']
scaler = StandardScaler()
data.loc[:,feat] = scaler.fit_transform(data.loc[:,feat]) 
data.loc[:,log_feat] = scaler.fit_transform(log_scaler(data.loc[:,log_feat]))
#data.loc[:,['Words']] = scaler.fit_transform(log_scaler(data.loc[:,['Words']]))
fml =  'Words~nodes+density+assortativity+local_cluster+global_cluster+avg_degree+max_degree+large_com+betweenness+degree+eigen+nodes*density'
fml = 'Words~nodes*density'

md = smf.mixedlm(fml, data, groups=data["Subreddits"])
mdf = md.fit()
print(mdf.summary())

model = sm.GLM.from_formula(fml, data, family=sm.families.Poisson())
results = model.fit()
print(results.summary())




from sklearn import decomposition
X = data.loc[:,features]
pca = decomposition.PCA(n_components=5)
pca.fit(X)
X = pca.transform(X)
pca.explained_variance_ratio_


from sklearn.manifold import TSNE,MDS
import umap

X = data[data['Time']=='2018_10'].loc[:,features]
label = data[data['Time']=='2018_10'].loc[:,'Words']

tsne = TSNE(n_components=2,perplexity=10)
Y = tsne.fit_transform(X)


plt.scatter(Y[:,0],Y[:,1])
plt.scatter(Y[number_of_subr:,0],Y[number_of_subr:,1])

reducer = umap.UMAP()
Y = reducer.fit_transform(X)



fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(16, 4))
ax0.set_title("Words")
_ = data["Words"].hist(bins=30, log=True, ax=ax0)
ax1.set_title("nodes")
_ = data["nodes"].hist(bins=30, log=True, ax=ax1)
ax2.set_title("Density")
_ = data["density"].hist(bins=30, log=True, ax=ax2)


fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(16, 4))
ax0.set_title("Assortativity")
_ = data["assortativity"].hist(bins=30, log=True, ax=ax0)
ax1.set_title("Local_cluster")
_ = data["local_cluster"].hist(bins=30, log=True, ax=ax1)
ax2.set_title("Global_cluster")
_ = data["global_cluster"].hist(bins=30, log=True, ax=ax2)


fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(16, 4))
ax0.set_title("Avg_degree")
_ = data["avg_degree"].hist(bins=30, log=True, ax=ax0)
ax1.set_title("Max_degree")
_ = data["max_degree"].hist(bins=30, log=True, ax=ax1)
ax2.set_title("large_com")
_ = data["large_com"].hist(bins=30, log=True, ax=ax2)


fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(16, 4))
ax0.set_title("degree")
_ = data["degree"].hist(bins=30, log=True, ax=ax0)
ax1.set_title("betweenness")
_ = data["betweenness"].hist(bins=30, log=True, ax=ax1)
ax2.set_title("pagerank")
_ = data["pagerank"].hist(bins=30, log=True, ax=ax2)
'''
