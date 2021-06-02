import rapidjson as json
import os
from tqdm import tqdm
from itertools import chain



path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/words'
subreddits = os.listdir(path)

paths = [os.path.join(path,s) for s in subreddits]

vocab = []

for p in tqdm(paths):
    try:
        sub_dict = json.load(open(p,'r'))
        items = set(chain.from_iterable(list(v.keys()) for v in sub_dict.values()))
            
        vocab.append(list(items))
    except:
        print(p)
 
print('Done')
vocab = list(set(chain.from_iterable(vocab)))   
with open('./summaries/total_vocab','w') as f:
    for w in vocab:
        f.write('%s\n'%(w))