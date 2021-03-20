#%%
import numpy as np 
import pandas as pd 
import cremerlab.hplc 
import matplotlib.pyplot as plt 
import imp 
imp.reload(cremerlab.hplc)

data = pd.read_csv('../exploratory_data/out/2021-03-18_NC_base001_test.csv', comment='#')
data.head()

# %%
fig, ax = plt.subplots(1, 1)
ax.plot(data['time_min'], data['intensity_mV'])
# %%
