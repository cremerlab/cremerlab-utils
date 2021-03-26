#%%

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cremerlab.hplc
import glob
import imp
imp.reload(cremerlab.hplc)
#%%

files = glob.glob('../exploratory_data/2021-03-25_test/*.txt')
dfs = cremerlab.hplc.convert(files)

#%%

valid_dfs = [dfs[0], dfs[4], dfs[6], dfs[12]]
peak_dfs = []
chroms = []
for i, d in enumerate(valid_dfs):
    chrom = cremerlab.hplc.Chromatogram(dataframe=d, time_window=[9, 17])
    peaks = chrom.quantify(rel_height=0.999)
    chrom.show()
    peaks['idx'] = i+1
    peak_dfs.append(peaks)
    chroms.append(chrom)
#%%
peak_df = pd.concat(peak_dfs, sort=False)
# Assign the peak ids
peak_df.loc[(peak_df['retention_time'] < 15), 'peak_idx'] = 1
peak_df.loc[(peak_df['retention_time'] > 15) & 
            (peak_df['retention_time'] < 15.5), 'peak_idx'] = 2
peak_df.loc[(peak_df['retention_time'] > 15.5), 'peak_idx'] = 3

fig, ax = plt.subplots(2, 1, figsize=(6, 6))
for g, d in peak_df.groupby(['peak_idx']):
    ax[0].plot(d['idx'].values, d['area'], 'o--', label=f'file # {g}')
    ax[1].plot(d['idx'].values, d['retention_time'], 'o--', label=f'file # {g}')

ax[0].legend()
ax[0].set_title('peak areas')
ax[1].set_title('peak retention time')
ax[0].set_ylabel('integrated area [mV]')
ax[1].set_ylabel('retention time [min]')
for a in ax:
    a.set_xlabel('order measured')
plt.tight_layout()
plt.savefig('/Users/gchure/Desktop/2021-03-25_test.pdf')

#%%
ret_times = [11.20, 15.14, 15.88, 11.18, 15.25, 15.79, 11.164, 15.12, 15.82, 11.15, 15.118, 15.794]
areas = [512092, 1282506,305119, 1242558, 3106865, 800834, 612356, 1540182, 368248, 682092, 1751140, 424883]
peak_ids = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
order = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
peak_table = pd.DataFrame([])
peak_table['retention_time'] = ret_times
peak_table['area'] = areas 
peak_table['peak_idx'] = peak_ids
peak_table['idx'] = order


fig, ax = plt.subplots(2, 1, figsize=(6, 6))
for g, d in peak_table.groupby(['peak_idx']):
    ax[0].plot(d['idx'].values, d['area'], 'o--', label=f'peak # {g}')
    ax[1].plot(d['idx'].values, d['retention_time'], 'o--', label=f'peak # {g}')

ax[0].legend()
ax[0].set_title('peak areas')
ax[1].set_title('peak retention time')
ax[0].set_ylabel('integrated area [mV]')
ax[1].set_ylabel('retention time [min]')
for a in ax:
    a.set_xlabel('replicate')
    a.set_xticks([1, 2, 3, 4])
plt.tight_layout()
plt.savefig('/Users/gchure/Desktop/2021-03-25_shimadzu_peak_table.pdf')
#%%
fig, ax = chroms[0].show()
plt.savefig('/Users/gchure/Desktop/2021-03-25_representative_chrom.pdf')
#%%
# Load the files
_files = ['ASCIIData013.txt', 'ASCIIData015.txt', 'ASCIIData017.txt']
raw_files = [f'../exploratory_data/2021-03-23_baseline_test/{f}' for f in _files]
# raw_files = glob.glob('../exploratory_data/2021-03-23_baseline_test/.txt')
dfs = cremerlab.hplc.convert(raw_files)

#%%

processed_files = glob.glob('../exploratory_data/2021-03-23_baseline_test/out/*.csv')
peaks_df = []
file = [13, 15, 17]
_chroms = []
for i, p in enumerate(processed_files):
    chrom = cremerlab.hplc.Chromatogram(csv_file=p, time_window=[13, 17])
    _chroms.append(chrom)
    peaks = chrom.quantify(buffer=100)
    peaks['file_idx'] = file[i]
    peaks_df.append(peaks)
    chrom.show()
_peaks = pd.concat(peaks_df, sort=False)
_peaks

#%%
# Load the files
raw_files = glob.glob('../exploratory_data/2021-03-17_calibration/*.txt')
dfs = cremerlab.hplc.convert(raw_files)

#%%

processed_files = glob.glob('../exploratory_data/2021-03-17_calibration/out/*sample1_*.csv')[:-3]
concs = [5, 10, 15, 20, 25, 30]
chroms_ =[]
peaks_ = []
for i, p in enumerate(processed_files):
    chrom = cremerlab.hplc.Chromatogram(csv_file=p, time_window=[13, 27])
    peaks = chrom.quantify()
    peaks['conc'] = concs[i]
    peaks_.append(peaks)
    chroms_.append(chrom)
peak_df = pd.concat(peaks_, sort=False)

peak_df['retention_time'] = np.round(peak_df['retention_time'].values, decimals=1)
peak_df = peak_df[peak_df['retention_time'] > 15]

# %%

for c in chroms_:
    c.show()
# %%
