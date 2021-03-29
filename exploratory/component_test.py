#%%
import numpy as np 
import pandas as pd 
import glob
import scipy.stats
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()
import tqdm
import altair as alt 
import cremerlab.hplc 

#%%
files = glob.glob('./exploratory_data/2021-03-25_component_calibration/*.txt')
cremerlab.hplc.convert(files)

files = glob.glob('./exploratory_data/2021-03-25_component_calibration/out/*.csv')
names = [f.split('/')[-1] for f in files]
chroms = []
peak_dfs = []
for i, f in enumerate(tqdm.tqdm(files)):
    fname = f.split('/')[-1]
    # Parse the file names
    if '_' in fname:
        name, ind = fname.split('_')
        ind = int(ind.split('.csv')[0]) + 2
    else:
        name = fname.split('.csv')[0]
        ind = 1
    if '001' in name:
        name = name.split('001')[0]

    # Load and quantify
    chrom = cremerlab.hplc.Chromatogram(f, time_window=[1, 20])
    peaks = chrom.quantify()
    chroms.append(chrom)
    peaks['compound'] = name 
    peaks['replicate'] = ind 
    peak_dfs.append(peaks)

peak_df = pd.concat(peak_dfs, sort=False)
# %%
fig, ax = plt.subplots(1,1, figsize=(8, 6))
ax = [ax]
for a in ax:
    a.set_xlabel('replicate')
    a.set_xticks([1, 2, 3])
    a.set_ylabel('integrated signal [mV]')

# iter = 0
for g, d in peak_df.groupby(['compound']):
    for _g, _d in d.groupby(['peak_idx']):
        _d.sort_values(by='replicate', inplace=True)
        ax[0].plot(_d['replicate'], _d['area'], 'o--', label=f'{g}, peak {_g + 1}')  
    # ax[iter].legend(title='peak #')
    # ax[iter].set_title(g)
    # iter += 1
ax[0].legend(bbox_to_anchor=(1,1))
plt.tight_layout()
plt.savefig('/Users/gchure/Desktop/2021-03-25_component_test.pdf')


#%%
for g, d in peak_df.groupby(['compound']):

    if len(d['peak_idx'].unique() > 1):
        fig, ax = plt.subplots(1, 2, figsize=(6, 4))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax = [ax]
    iter = 0 
    for _g, _d in d.groupby(['peak_idx']):
        _agg = _d.groupby()
        for __g, __d in _d.groupby(['replicate']):
            fit = __d['amplitude'].values[0] * scipy.stats.skewnorm(__d['skew'].values[0], __d['retention_time'].values[0], __d['scale'].values[0]).pdf(time_range)
            ax[iter].plot(time_range, fit, 'k-', lw=1)
            ax[iter].fill_between(time_range, 0, fit, alpha=0.5)
        iter += 1 

# %%
files = glob.glob('./exploratory_data/2021-03-25_component_calibration/out/*PO4*.csv')
import imp 
imp.reload(cremerlab.hplc)
chroms, peaks, fig, ax = cremerlab.hplc.batch_process(files[:-1], show_viz=True, time_window=[0, 25])





# %%
