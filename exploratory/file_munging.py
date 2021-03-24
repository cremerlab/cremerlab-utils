#%%
import pandas as pd 
import cremerlab.hplc
import imp 
import glob
imp.reload(cremerlab.hplc)

# %%
files = glob.glob('../exploratory_data/2021-03-17_calibration/*.txt')
dfs, metadata = cremerlab.hplc.convert(files, save_suffix='test',metadata=True)

# %%
