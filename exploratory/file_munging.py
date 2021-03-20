#%%
import pandas as pd 
import cremerlab.hplc
import imp 
import glob
imp.reload(cremerlab.hplc)

# %%
files = glob.glob('../exploratory_data/*.txt')
dfs, metadata = cremerlab.hplc.convert(files, save_suffix='test',metadata=True)

# %%
