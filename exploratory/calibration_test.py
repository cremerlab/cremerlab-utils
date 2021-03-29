#%%
import glob 
import pandas as pd
import numpy as np 
import imp 
import cremerlab.hplc
import matplotlib.pyplot as plt
imp.reload(cremerlab.hplc)

raw_files = glob.glob('./exploratory_data/2021-03-26_calibration/*.txt')
cremerlab.hplc.convert(raw_files)

#%%
imp.reload(cremerlab.hplc)
calibration_files = glob.glob('./exploratory_data/2021-03-26_calibration/out/*calibration*.csv')
files = [str(f) for f in np.sort(calibration_files)]
chroms, peaks, fig, ax = cremerlab.hplc.batch_process(files, 
                                                      time_window=[13, 28],
                                                      show_viz=True)
plt.savefig('/Users/gchure/Desktop/2021-03-26_calibration_chromatograms.pdf')

peaks['concentration_mM'] = [float(s.split('_')[-1].split('mM')[0]) for s in peaks['sample'].values]

# Assign names to peaks
#%%
import scipy.stats
import seaborn as sns 
sns.set()
peaks.loc[(peaks['retention_time'] > 15) & (peaks['retention_time'] < 16), 'compound'] = 'glucose'
peaks.loc[(peaks['retention_time'] > 20) & (peaks['retention_time'] < 22.5), 'compound'] = 'glycerol'
peaks.loc[(peaks['retention_time'] > 24) & (peaks['retention_time'] < 26), 'compound'] = 'acetate'
peaks.dropna(inplace=True)
fig, ax = plt.subplots(1,1)
stats = {}
conc_range = np.linspace(0, 35, 100)

_colors = sns.color_palette('deep')
colors = {'glucose': _colors[0], 'glycerol':_colors[1], 'acetate':_colors[2]}
for g, d in peaks.groupby('compound'):
    popt = scipy.stats.linregress(d['concentration_mM'], d['area'])
    stats[g] = {'slope': popt[0], 'intercept':popt[1]} 
    _fit = popt[1] + popt[0] * conc_range
    ax.plot(d['concentration_mM'], d['area'], 'o', label=g, color=colors[g])
    ax.plot(conc_range, _fit, label='__nolegend__', color=colors[g])
ax.set_xlabel('concentration [mM]')
ax.set_ylabel('integrated signal [mV]')
ax.legend()
plt.savefig('/Users/gchure/Desktop/2021-03-26_calibration_curve.pdf')

#%%
strain = 'REL'
import imp 
imp.reload(cremerlab.hplc)
turnover_files = glob.glob(f'./exploratory_data/2021-03-26_calibration/out/*{strain}*.csv')
files = [str(f) for f in np.sort(turnover_files)]

chroms, peaks, fig, ax = cremerlab.hplc.batch_process(files, 
                                                      time_window=[13, 28],
                                                      show_viz=True)
plt.savefig(f'/Users/gchure/Desktop/2021-03-26_{strain}_chromatogram.pdf')



peaks.loc[(peaks['retention_time'] > 15.5) & (peaks['retention_time'] < 16), 'compound'] = 'glucose'
peaks.loc[(peaks['retention_time'] > 20) & (peaks['retention_time'] < 22.5), 'compound'] = 'glycerol'
peaks.loc[(peaks['retention_time'] > 24) & (peaks['retention_time'] < 26), 'compound'] = 'acetate'
peaks['timepoint'] = [int(f.split('_')[-1]) for f in peaks['sample'].values]

#%%

# REL
peaks.loc[peaks['timepoint']==1, 'od'] = 0.113
peaks.loc[peaks['timepoint']==2, 'od'] = 0.207
peaks.loc[peaks['timepoint']==3, 'od'] = 0.329
peaks.loc[peaks['timepoint']==4, 'od'] = 0.392
peaks.loc[peaks['timepoint']==5, 'od'] = 0.650

fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
iter = 0
for g, d in peaks.groupby(['compound']):
    conc = (d['area'].values - stats[g]['intercept']) / stats[g]['slope']
    ax[iter].plot(d['od'], conc, 'o--', label=g)
    ax[iter].set_title(g)
    iter += 1
for i in range(iter):
    ax[i].set_xlabel('optical density')
    ax[i].set_ylabel('concentration [mM]')

    # ax.legend()
plt.tight_layout()
plt.savefig('/Users/gchure/Desktop/REL606_glucose_acetate_turnover.pdf')
# %%
_chrom = chroms[chroms['sample'] == '2021-03-18_REL606_turnover_timepoint_005']
import scipy.signal
# fig, ax = plt.subplots(1, 1)
# peaks = scipy.signal.find_peaks_cwt(_chrom['intensity_mV'].values, np.arange(10, 100))
                                    
# ax.plot(_chrom['time_min'], _chrom['intensity_mV'])
# ax.plot(_chrom['time_min'].values[peaks], _chrom['intensity_mV'].values[peaks], 'o')

_norm = (_chrom['intensity_mV'].values - _chrom['intensity_mV'].min()) / (_chrom['intensity_mV'].max() - _chrom['intensity_mV'].min())

peaks, _ = scipy.signal.find_peaks(_norm, prominence=1E-4)
# scipy.signal.peak_prominences(_norm, peaks)
fig, ax = plt.subplots(1, 1)
ax.plot(_chrom['time_min'], _chrom['intensity_mV'])
ax.plot(_chrom['time_min'].values[peaks], _chrom['intensity_mV'].values[peaks], 'o')
ax.set_yscale('log')

# %%
# NCM
peaks.loc[peaks['timepoint']==1, 'od'] = 0.109
peaks.loc[peaks['timepoint']==2, 'od'] = 0.276
peaks.loc[peaks['timepoint']==3, 'od'] = 0.383
peaks.loc[peaks['timepoint']==4, 'od'] = 0.457
peaks.loc[peaks['timepoint']==5, 'od'] = 0.740

fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
iter = 0
for g, d in peaks.groupby(['compound']):
    conc = (d['area'].values - stats[g]['intercept']) / stats[g]['slope']
    ax[iter].plot(d['od'], conc, 'o--', label=g)
    ax[iter].set_title(g)
    iter += 1
for i in range(iter):
    ax[i].set_xlabel('optical density')
    ax[i].set_ylabel('concentration [mM]')

    # ax.legend()
plt.tight_layout()
plt.savefig('/Users/gchure/Desktop/NCM3722_glucose_acetate_turnover.pdf')
# %%
_chrom = chroms[chroms['sample'] == '2021-03-18_REL606_turnover_timepoint_005']
import scipy.signal
# fig, ax = plt.subplots(1, 1)
# peaks = scipy.signal.find_peaks_cwt(_chrom['intensity_mV'].values, np.arange(10, 100))
                                    
# ax.plot(_chrom['time_min'], _chrom['intensity_mV'])
# ax.plot(_chrom['time_min'].values[peaks], _chrom['intensity_mV'].values[peaks], 'o')

_norm = (_chrom['intensity_mV'].values - _chrom['intensity_mV'].min()) / (_chrom['intensity_mV'].max() - _chrom['intensity_mV'].min())

peaks, _ = scipy.signal.find_peaks(_norm, prominence=1E-4)
# scipy.signal.peak_prominences(_norm, peaks)
fig, ax = plt.subplots(1, 1)
ax.plot(_chrom['time_min'], _chrom['intensity_mV'])
ax.plot(_chrom['time_min'].values[peaks], _chrom['intensity_mV'].values[peaks], 'o')
ax.set_yscale('log')

# %%
