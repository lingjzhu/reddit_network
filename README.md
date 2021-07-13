## Code for *The structure of online social networks modulates the rate of lexical change*
Published in NAACL-HLT 2021 [(Link)](https://aclanthology.org/2021.naacl-main.178/). 


### Data
  - List of subreddits: samples.csv
  - Neologisms: neologisms.csv

### Hardware
  - The raw data is around 2TB;
  - The generated graph data are around 300GB;
  - The generated user and vocabulary lists, and immediate data may take up a few hundred GBs;
  - We use 8~10 cores and 120 GB of memory to parallelize many of the computations. For example, extracting all graphs may take more than a week using a single core but can be completed in less than 2 days using 8 cores;
  - It should take less than two weeks to complete all the computation in our case. 

### Steps
  - Download data -> *download_map.py*
  - Get all users and vocabulary -> *get_words_users_map.py*
  - Intra-community graph construction -> *graph_map.py*
  - Inter-community graph contruction -> *intercomm_graph.py*
  - Generating random baseline graphs -> *randomgraph_baseline.py*
  - Extract graph features -> *graph_stats.py*
  - Poisson regression -> *poisson_preprocessing.py* and *poisson_pred_pca.py*
  - Survival analysis -> *survival_processing.py* and *survival_analysisi_pca.py*

You can cite this article as
```
@inproceedings{zhu-jurgens-2021-structure,
    title = "The structure of online social networks modulates the rate of lexical change",
    author = "Zhu, Jian  and
      Jurgens, David",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.178",
    pages = "2201--2218"
}
```
