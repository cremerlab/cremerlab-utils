import pandas as pd 
import numpy as np
from io import StringIO
import scipy.signal
import scipy.optimize
import tqdm
import os 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def scrape_metadata(file, delimiter=','):
    """
    Scrapes the sample information from the output of a Shimadzu HPLC ASCII 
    file and returns a dictionary of the metadata entries. 

    Parameters
    ----------
    file : str 
        The contents of the file as a string with newline characters (`\n`) present. 
    delimiter: str 
        The delimeter used in the file. If  tab-delimited, use `\t`.
    Returns
    -------
    metadata_dict : dictionary
        A dictionary of the metadata listed under the 'Sample Information', excluding 
        ISTD Amounts.

    Notes
    -----
    This function assumes that the file contains metadata fields `[Sample Information]`
    followed by `[Original Files]`. If either `[Sample Information]` or `[Original Files]`
    fields are missing or have other names, a ValueError exception is thrown. 

    Raises
    ------
    TypeError
        Raised if file is not of type `str`
    ValueError:
        Raised if `[Sample Information]`, `[Original Files]`, `\\n`, or delimiter
        is not present in file
    """
    # Make sure the 'file' provided is a string and has necessary fields. 
    if type(file) != str:
        raise TypeError(f'Argument `file` must be a string. type {type(file)} was provided.')
    if '\n' not in file:
        raise ValueError(f'Newline characters (`\\n`) are not in file, but must be for proper parsing.')
    if ('[Sample Information]' not in file) | ('[Original Files]' not in file):
        raise ValueError('`[Sample Information]` or `[Original Files]` field is missing.')
    if delimiter not in file:
        raise ValueError(f'Delimiter {f} not present in the file.')

    # Get the sample information and split by newline
    metadata = file.split('[Sample Information]\n')[1].split('[Original Files]\n')[0]
    
    # Set up the dictionary and loop through lines
    metadata_dict = {}
    for line in metadata.split('\n'):

        # Split by the delimiter.
        entry = line.split(delimiter)
        
        # Add the entry to the dictionary if longer than 1 and does not 
        # contain ISTD Amount.
        if (len(entry) > 1) & ('ISTD' not in entry[0]):
            metadata_dict[entry[0]] = entry[1]
    return metadata_dict

def scrape_chromatogram(file, detector='B', delimiter=',', metadata=True):
    """
    Scrapes the chromatogram for a given detector from a Shimadzu HPLC
    ASCII output.

    Parameters
    ----------
    file : str 
        The contents of the file as a string with newline characters (`\n`) present. 
    detector: str, 'A' or 'B'
        The name of the detector in the file. Default is `B`. Note that only that only 
        'A' or 'B' is an acceptable input and not the complete detector name such as 
        'LC Chromatogram(Detector B-Ch1)'.
    delimiter: str 
        The delimeter used in the file. If  tab-delimited, use `\t`.
    metadata : bool
        If `true`, a dictionary with the metadata about the detector is returned. 
        Default is True. 

    Returns
    -------
    chrom : pandas DataFrame
        A tidy pandas DataFrame with two columns -- `time_minutes` and `intensity_mV`
    metadata_dict : dictionary
        A dictionary of the metadata associated with the desired detector channel.
        if `metadata` is not `True`, this is not returned. 

    Notes
    -----
    This function assumes that the detector name follows the convention 
    `[LC Chromatogram(Detector A/B-ChX]` where `A/B` is the detector label 
    and `X` is the channel number. 

    Raises
    ------
    TypeError
        Raised if file, detctor, or delimiter is not of type `str`.
    ValueError:
        Raised if `[LC Chromatogram(Detector`, `\\n`, or delimiter is not present in file.
        Also raised if `detector.upper()` is not `A` or `B`
    """
    # Do type checks. 
    for arg in (file, detector, delimiter):
        if type(arg) is not str:
            raise TypeError(f'Type of `{arg}` must be `str`. Type {type(arg)} was provided.')
    for st in ['[LC Chromatogram(Detector', '\n', delimiter]:
        if st not in file:
            raise ValueError(f'Pattern {st} not present in file.')
    if (detector.upper() != 'A') & (detector.upper() != 'B'):
        raise ValueError(f'Detector must be `A` or `B`. String of `{detector.upper()}` was provided.')

    # Parse and split the file to get the chromatogram
    chrom = '\n'.join(file.split(
                    f'[LC Chromatogram(Detector {detector.upper()}-Ch'
                    )[1].split('\n')[1:-2])

    # Read into as a dataframe.
    df = pd.read_csv(StringIO(chrom), delimiter=delimiter, skiprows=6)

    # Rename the columns
    df.columns = ['time_min', 'intensity_mV']
    df['detector'] = detector

    # Determine if the metadata should be scraped and returned
    if metadata:
        metadata_dict = {}
        for line in chrom.split('\n')[:6]:
            entry = line.split(delimiter)
            if len(entry) > 0:
                metadata_dict[entry[0]] = entry[1]
        out = [df, metadata_dict]
    else:
        out = df
    return out

def convert(file_path, detector='B', delimiter=',', metadata=False,  save=True,
            output_dir=None, save_prefix=None, save_suffix=None, verbose=True):
    """
    Reads the ASCII output from a Shimadzu HPLC and returns a DataFrame of the
    chromatogram. Converted files can also be saved to disk with human-readable 
    metadata. 

    Parameters
    ----------
    file_path : str or list of str
        The file path of the ASCII file to read. Multiple files can be provided 
        as a list of file paths. 
    detector : str or list of str, ['A' or 'B']
        The detector channel chromatogram to return. Must be either 'A' or 'B' 
        or both, if provided as a list. Default is just 'B'.
    delimiter: str 
        The delimiter used in the file. If  tab-delimited, use `\t`. Default is 
        a comma, `,`.
    metadata : bool
        If True, the file  and instrument metadata are returned as a dictionary.
    save: bool 
        If `True`, the dataframe is saved as a csv with the metadata as a 
        header prefixed with '#'. Default is `False`.
    output_dir : str or None
        The output directory if the dataframe are to be saved to disk. If 
        `None` and `save = True`, the dataframe will be saved in the directory 
        of the `file` in an `output` folder. 
    save_prefix : str
        A prefix to append to the file name if saved to disk. If None, 
        saved filed will just be `SAMPLE_NAME.csv`.       
    save_suffix : str
        A suffix to append to the file name if saved to disk. If None, 
        saved filed will just be `SAMPLE_NAME.csv`.       
    verbose : bool 
        If True a progress bar will print if there's more than 10 files to 
        process.
    Returns
    -------
    df : pandas DataFrame or list of pandas DataFrames.
        The chromatograms for each file and detector pattern.   
    metadata_dict : dict 
        A dictionary of the sample and detector metadata. This is only 
        returned if `metadata=True`. 
    """

    # Determine the size of the  
    if type(file_path) is str:
        file_path = [file_path]

    # Determine if there should be a verbose output
    if (verbose is True) & (len(file_path) >= 10):
        iterator = enumerate(tqdm.tqdm(file_path, desc='Converting ASCII output.'))
    else:
        iterator = enumerate(file_path)
    
    # If save, determine the output directory
    #TODO: Make sure this works on a windows system
    if (save is True) & (output_dir is None):
        output_path = '/'.join(file_path[0].split('/')[:-1])
        if os.path.isdir(f'{output_path}/out') == False:
            os.mkdir(f'{output_path}/out')
        output_path += '/out'

    # Instantiate 
    dfs = []
    metadata_dict = [] 
    for i, f in iterator:
        with open(f, 'r') as file:
            raw_file = file.read()
            # Get the file metadata 
            file_metadata = scrape_metadata(raw_file, delimiter=delimiter)

            # Get chromatogram
            if type(detector) == str:
                detector = [detector]
            chroms = []
            chrom_metadata = {}
            chrom_file_metadata = []
            for d in detector:

                # Parse the chromatogram
                _chrom, _chrom_metadata = scrape_chromatogram(raw_file, 
                                                            detector=d, 
                                                            delimiter=delimiter)    
                chroms.append(_chrom)
                chrom_metadata[f'Detector {d}'] = _chrom_metadata

                # Generate the metadata for file saving
                _metadata = f"""#
# Detector {d.upper()}
# ---------------------
# Acquisition Interval: {_chrom_metadata['Interval(msec)']} ms
# Intensity Units: {_chrom_metadata['Intensity Units']}
# Intensity Multiplier: {_chrom_metadata['Intensity Multiplier']}
#
"""
                chrom_file_metadata.append(_metadata)
            if len(detector) == 1: 
                chrom = chroms[0]
            else:
                chrom = pd.concat(chrom, sort=False)
            dfs.append(chrom)
            metadata_dict.append(chrom_metadata)
            if type(detector) == str:
                detector = [detector]

            # Assemble the comment file 
            header = f"""#
# {file_metadata['Sample Name']}
# ------------------------------
# Acquired: {file_metadata['Acquired']}
# Converted: {datetime.now().strftime('%H:%M -- %Y-%m-%d')}
# Vial: {file_metadata['Vial#']}
# Injection Volume: {file_metadata['Injection Volume']} uL
# Sample ID: {file_metadata['Sample ID']}
"""
            for m in chrom_file_metadata:
                header += m

            # Process prefix and suffix to the save file
            name = ''
            if save_prefix is not None:
                name += save_prefix + '_'
            name += file_metadata['Sample Name']
            if save_suffix is not None:
                name += '_' + save_suffix
            name += '.csv'

            if save:
                save_name = f'{output_path}/{name}'
                with open(save_name, 'a') as save_file:
                    save_file.write(header)

                    chrom.to_csv(save_file, index=False) 

    # Determine what to return and get outta here.
    if len(file_path) > 0:
        df = dfs[0]
        metadata_dict = metadata_dict[0]
    if metadata: 
        out = [df, metadata_dict]
    else:
        out = df
    return out


def approx_peak_integration(intensity, window=None):
    """
    Performs an approximate integration of the identified peaks
    """

class Chromatogram(object):
    """
    Base class for dealing with HPLC chromatograms
    """
    def __init__(self, csv_file=None, dataframe=None, time_window=None,
                    cols={'time':'time_min', 'intensity':'intensity_mV'},
                    csv_comment='#'):
        """
        Instantiates a chromatogram object on which peak detection and quantification
        is performed.

        Parameters
        ----------
        csv_file: str, optional
            The path to the csv file of the chromatogram to analyze. If None, 
            a pandas DataFrame of the chromatogram must be passed.
        dataframe : pandas DataFrame, optional
            a Pandas Dataframe of the chromatogram to analyze. If None, 
            a path to the csv file must be passed
        time_window: list [start, end], optional
            The retention time window of the chromatogram to consider for analysis.
            If None, the entire time range of the chromatogram will be considered.
        cols: dict, keys of 'time', and 'intensity', optional
            A dictionary of the retention time and intensity measurements 
            of the chromatogram. Default is 
            `{'time':'time_min', 'intensity':'intensity_mV'}`.
        csv_comment: str, optional
            Comment delimiter in the csv file if chromatogram is being read 
            from disk.
        """

        # Peform type checks and throw exceptions where necessary. 
        if (csv_file is None) & (dataframe is None):
            raise RuntimeError(f'Neither `csv_file` or `dataframe` is provided!')
        if (csv_file is not None) & (dataframe is not None):
            raise RuntimeError(f'Either `csv_file` or `dataframe` must be provided, not both.')
        if (csv_file is not None) & (type(csv_file) != str):
            raise TypeError(f'`csv_file` must be a str. Type `{type(csv_file)}` was provided.')
        if (dataframe is not None) & (type(dataframe) != pd.core.frame.DataFrame):
            raise TypeError(f'`dataframe` must be of type `pd.core.frame.Dataframe`. Type `{type(dataframe)}` was provided')
        if (time_window is not None):
            if type(time_window) != list:
                raise TypeError(f'`time_window` must be of type `list`. Type {type(time_window)} was proivided')
            if len(time_window) != 2:
                raise ValueError(f'`time_window` must be of length 2 (corresponding to start and end points). Provided list is of length {len(time_window)}.')

        # Load the chromatogram and necessary components to self. 
        if (csv_file is not None):
            dataframe = pd.read_csv(csv_file, comment='#')
        
        # Prune to time window
        if time_window is not None:
            self.df = dataframe[(dataframe[cols['time']] >= time_window[0]) & 
                              (dataframe[cols['time']] <= time_window[1])]
        else: 
            self.df = dataframe

        # Add column dict
        self.time_col = cols['time']
        self.int_col = cols['intensity']

        # Blank out vars that are used elsewhere
        self.window_df = None
        self.window_props = None
        self.peaks = None

    def _assign_peak_windows(self, prominence=0.01, rel_height=0.95, buffer=100):
        """
        Breaks the provided chromatogram down to windows of likely peaks. 

        Parameters
        ----------
        prominence : float,  [0, 1]
            The promimence threshold for identifying peaks. Prominence is the 
            relative height of the normalized signal relative to the local
            background. Default is 1%.
        rel_height : float, [0, 1]
            The relative height of the peak where the baseline is determined. 
            Default is 95%.
        buffer : positive int
            The padding of peak windows in units of number of time steps. Default 
            is 100 points on each side of the identified peak window.

        Returns
        -------
        windows : pandas DataFrame
            A Pandas DataFrame with each measurement assigned to an identified 
            peak or overlapping peak set. This returns a copy of the chromatogram
            DataFrame with  a column  for the local baseline and one column for 
            the window IDs. Window ID of -1 corresponds to area not assigned to 
            any peaks
        """
        for param, param_type in zip([prominence, rel_height, buffer], 
                                     [float, float, int]):
            if type(param) is not param_type:
                raise TypeError(f'Parameter {param} must be of type `{param_type}`. Type `{type(param)}` was supplied.') 
        if (prominence < 0) | (prominence > 1):
            raise ValueError(f'Parameter `prominence` must be [0, 1].')
        if (rel_height < 0) | (rel_height > 1):  
            raise ValueError(f'Parameter `rel_height` must be [0, 1].')
        if (buffer < 0):
            raise ValueError('Parameter `buffer` cannot be less than 0.')

        # Correct for a negative baseline 
        df = self.df
        min_int = df[self.int_col].min() 
        if min_int < 0:
           min_int = np.abs(min_int)    
        intensity = df[self.int_col] + min_int

        # Normalize the intensity 
        norm_int = (intensity - intensity.min()) / (intensity.max() - intensity.min())

        # Identify the peaks and get the widths and baselines
        peaks, _ = scipy.signal.find_peaks(norm_int, prominence=prominence)
        self.peaks_inds = peaks
        out = scipy.signal.peak_widths(norm_int, peaks, rel_height=rel_height)
        _, heights, left, right = out

        # Set up the ranges
        ranges = []
        for l, r in zip(left, right):
            if (l - buffer) < 0:
                l = 0
            else:
                l -= buffer
            if (r + buffer) > len(norm_int):
                r = len(norm_int)
            else:
                r += buffer
            ranges.append(np.arange(np.round(l), np.round(r), 1))

        # Identiy subset ranges and remove
        valid = [True] * len(ranges)
        for i, r1 in enumerate(ranges):
            for j, r2 in enumerate(ranges):
                if i != j:
                    if set(r2).issubset(r1):
                        valid[j] = False
        
        # Keep only valid ranges and baselines
        ranges = [r for i, r in enumerate(ranges) if valid[i] is True]
        baselines = [h for i, h in enumerate(heights) if valid[i] is True]

        # Copy the dataframe and return the windows
        window_df = df.copy(deep=True)
        window_df.sort_values(by=self.time_col, inplace=True)
        window_df['time_idx'] = np.arange(len(window_df))
        for i, r in enumerate(ranges):
            window_df.loc[window_df['time_idx'].isin(r), 'window_idx'] = int(i + 1)
            window_df.loc[window_df['time_idx'].isin(r), 'baseline'] = baselines[i]
        window_df.dropna(inplace=True) 

        # Convert this to a dictionary for easy parsing
        window_dict = {}
        for g, d in window_df.groupby('window_idx'):
            _peaks = [p for p in peaks if p in d['time_idx'].values]
            _dict = {'time_range':d[self.time_col].values,
                     'intensity': d[self.int_col].values,
                     'num_peaks': len(_peaks),
                     'amplitude': [d[d['time_idx']==p][self.int_col].values[0] for p in _peaks],
                     'location' : [d[d['time_idx']==p][self.time_col].values[0] for p in _peaks]
                     }
            window_dict[g] = _dict
        self.window_props = window_dict
        return window_df  

    def _compute_skewnorm(self, x, *params):
        R"""
        Computes the lineshape of a skew-normal distribution given the shape,
        location, and scale parameters

        Parameters
        ----------
        x : float or numpy array
            The time dimension of the skewnorm 
        params : list, [amplitude, loc, scale, alpha]
            Parameters for the shape and scale parameters of the skewnorm 
            distribution.
                amplitude : float; > 0
                    Height of the peak.
                loc : float; > 0
                    The location parameter of the distribution.
                scale : float; > 0
                    The scale parameter of the distribution.
                alpha : float; > 
                    THe skew shape parater of the distribution.

        Returns
        -------
        scaled_pdf : float or numpy array, same shape as `x`
            The PDF of the skew-normal distribution scaled with the supplied 
            amplitude.
        """
        amp, loc, scale, alpha = params
        return amp * scipy.stats.skewnorm(alpha, loc=loc, scale=scale).pdf(x)

    def _fit_skewnorms(self, x, *params):
        R"""
        Estimates the parameters of the distributions which consititute the 
        peaks in the chromatogram. 

        Parameters
        ----------
        x : float
            The time dimension of the skewnorm 
        params : list of length 4 x number of peaks, [amplitude, loc, scale, alpha]
            Parameters for the shape and scale parameters of the skewnorm 
            distribution. Must be provided in following order, repeating
            for each distribution.
                amplitude : float; > 0
                    Height of the peak.
                loc : float; > 0
                    The location parameter of the distribution.
                scale : float; > 0
                    The scale parameter of the distribution.
                alpha : float; > 
                    THe skew shape parater of the distribution.

        Returns
        -------
        out : float
            The evaluated distribution at the given time x. This is the summed
            value for all distributions modeled to construct the peak in the 
            chromatogram.
        """
        # Get the number of peaks and reshape for easy indexing
        n_peaks = int(len(params) / 4)
        params = np.reshape(params, (n_peaks, 4))
        out = 0
        
        # Evaluate each distribution
        for i in range(n_peaks):
            out += self._compute_skewnorm(x, *params[i])
        return out
        
    def _estimate_peak_params(self, verbose=True):
        R"""
        For each peak window, estimate the parameters of skew-normal distributions 
        which makeup the peak(s) in the window.  

        Parameters
        ----------
        verbose : bool
            If `True`, a progress bar will be printed during the inference.
        """ 
        if self.window_props is None:
            raise RuntimeError('Function `_assign_peak_windows` must be run first. Go do that.')
        if verbose:
            iterator = tqdm.tqdm(self.window_props.items(), desc='Fitting peaks...')  
        else:
            iterator = self.window_props.items()

        peak_props = {}
        for k, v in iterator:
            window_dict = {}
            # Set up the initial guess
            p0 = []
            for i in range(v['num_peaks']):
                p0.append(v['amplitude'][i])
                p0.append(v['location'][i]),
                p0.append(1) # scale parameter
                p0.append(0) # Skew parameter, starts with assuming Gaussian

            # Perform the inference
            popt, _ = scipy.optimize.curve_fit(self._fit_skewnorms, v['time_range'],
                                               v['intensity'], p0=p0, maxfev=int(1E4))

            # Assemble the dictionary of output 
            if v['num_peaks'] > 1:
                popt = np.reshape(popt, (v['num_peaks'], 4)) 
            else:
                popt = [popt]
            for i, p in enumerate(popt):
                window_dict[f'peak_{i + 1}'] = {
                            'amplitude': p[0],
                            'retention_time': p[1],
                            'std_dev': p[2],
                            'alpha': p[3]}
            peak_props[k] = window_dict
        self.peak_props = peak_props
        return peak_props


               

    def show(self):
        """
        Displays the chromatogram with mapped peaks if available.
        """
        sns.set()
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.set_xlabel(self.time_col)
        ax.set_ylabel(self.int_col)

        # Show the raw chromatogram
        if self.window_df is None:
            ax.plot(self.df[self.time_col], self.df[self.int_col], '-', lw=2)
        
        else:
            for g, d in self.window_df.groupby(['window_idx']):
                ax.plot(d[self.time_col], d[self.int_col], label=int(g))

            leg = ax.legend(title='window ID', bbox_to_anchor=(1, 1))
    

