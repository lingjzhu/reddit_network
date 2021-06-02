import ujson as json
import os
from tqdm import tqdm

path = '/gpfs/accounts/lingjzhu_root/lingjzhu1/lingjzhu/graphs'
files = os.listdir(path)

files = ['baseball']
for f in tqdm(files):
    try:
        d = json.load(open(os.path.join(path,f),'r'))
    except:
        print('invalid %s'%(f))