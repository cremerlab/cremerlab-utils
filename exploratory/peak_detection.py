#%%
import numpy as np 
import pandas as pd
import cremerlab.hplc 
import matplotlib.pyplot as plt 
import scipy.signal
import scipy.special
import imp 
imp.reload(cremerlab.hplc)

start = 0
end = 45 * 60
step =  0.5
time = np.arange(start, end, step) 

# Define peaks and width
intensities = np.array([1700, 4000, 2000, 1000, 100, 3000, 1000, 1000, 3000])
locs = 60 * np.array([2, 10, 15, 20, 25, 30, 31.5, 37, 38.5 ])
scales =np.array([ 25, 15, 20, 30, 45, 20, 30, 30, 20])
ints = np.empty((len(locs), len(time)))

for i, loc in enumerate(locs):
    intensity = intensities[i]
    sigma = scales[i]
    signal = scipy.stats.norm(loc, sigma).pdf(time)
    signal *= signal.max()**-1
    ints[i, :] = intensity * signal

signal = np.sum(ints, axis=0)

# Compute the derivative of the signal
test_df = pd.DataFrame([])
test_df['intensity_mV'] = signal
test_df['time_min'] = time / 60

chrom = cremerlab.hplc.Chromatogram(dataframe=test_df)
window_df = chrom._assign_peak_windows()
chrom._estimate_peak_params()

#%%
# Compute a normalized intensity
# test_df = pd.read_csv('../exploratory_data/out/2021-03-17_NC_glucose_medium001_test.csv', comment='#')
test_df['norm_int'] = (test_df['intensity_mV'].values - test_df['intensity_mV'].min())/\
                      (test_df['intensity_mV'].max() - test_df['intensity_mV'].min())

# Add a time idx
# test_df['time_idx'] = np.arange(len(test_df), 1)

# find the peaks 
peaks, _= scipy.signal.find_peaks(test_df['norm_int'], prominence=0.01)
#%%
# Get the base positions and heights 
widths, heights, left, right = scipy.signal.peak_widths(test_df['norm_int'].values, 
                                                        peaks, rel_height=0.95)
                                                        
# Find nonoverlapping peak bases
buffer = 50 # in seconds
dt = 0.5
buffer *= dt**-1
ranges = [np.arange(np.round(l - buffer), np.round(r + buffer), 1) for l, r in zip(left, right)]
valid = [True] * len(ranges)
for i, r1 in enumerate(ranges):
    for j, r2 in enumerate(ranges):
        if i != j:
            if set(r2).issubset(r1):
                valid[j] = False
ranges = [r for i, r in enumerate(ranges) if valid[i] is True]
baselines = [h for i, h in enumerate(heights) if valid[i] is True]

fig, ax = plt.subplots(1, 1)
ax.plot(test_df['time_min'], test_df['intensity_mV'])
loc = 200
for i, r in enumerate(ranges):
    ax.plot(r / 120, loc * np.ones(len(r)), 'k-')
    loc += 100
#%%
# Chop up the dataframe into the mapped peaks
test_df_copy = test_df.copy(deep=True)
test_df_copy['idx'] = np.arange(len(test_df))
for i, r in enumerate(ranges):
    test_df_copy.loc[test_df_copy['idx'].isin(r), 'window'] = int(i + 1)
    test_df_copy.loc[test_df_copy['idx'].isin(r), 'baseline'] = baselines[i]
test_df_copy.dropna(inplace=True)

#%%
fig, ax = plt.subplots(1, 1)
for g, d in test_df_copy.groupby('window'):
    ax.plot(d['time_min'], d['intensity_mV'])

#%%
win = test_df_copy[test_df_copy['window'] == 4]
win['norm_int'] = win['intensity_mV']

def compute_gauss(x, amp, loc, scale, alpha):
    return amp * scipy.stats.skewnorm(alpha, loc=loc, scale=scale).pdf(x)

def gaussian_mix(x, *params):
    n_peaks = int(len(params)/4)
    params = np.reshape(params, (n_peaks, 4))
    out = 0
    for i in range(n_peaks):
        # out += amp * 2 * _norm.pdf(x) * _norm.cdf(alpha * x)
        out += compute_gauss(x, *params[i])
    return out

_amps = [win[win['idx']==p]['intensity_mV'].values[0] for p in peaks if p in win['idx']]
_locs = [win[win['idx']==p]['time_min'].values[0] for p in peaks if p in win['idx']]
_sigs = [1 for _ in range(len(_amps))] 
p0 = []

for i in range(len(_amps)):
    p0.append(_amps[i])
    p0.append(_locs[i])
    p0.append(_sigs[i])
    p0.append(0)
popt, cov = scipy.optimize.curve_fit(gaussian_mix, win['time_min'].values, win['intensity_mV'].values, p0=(p0,),
                                    maxfev=int(1E4))

# # Plot the fitted gaussian

fit1 =  compute_gauss(win['time_min'].values, popt[0], popt[1], popt[2], popt[3])
# fit2 =  compute_gauss(win['time_min'].values, *popt[4:])
# mix = fit1 + fit2

#%%
fig, ax = plt.subplots(1,1)

ax.plot(win['time_min'], win['intensity_mV'], 'k-', lw=2)
ax.fill_between(win['time_min'], 0, fit1, alpha=0.5)
# ax.fill_between(win['time_min'], 0, fit2, alpha=0.5)
# ax.plot(win['time_min'], mix, 'c--')
#%%
