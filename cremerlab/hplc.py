import pandas as pd 
import numpy as np
from io import StringIO
import shutil
import scipy.signal
import scipy.optimize
import scipy.special
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
        The contents of the file as a string with newline characters (`\\n`) present. 
    delimiter: str 
        The delimeter used in the file. If  tab-delimited, use `\\t`.
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
        The contents of the file as a string with newline characters (`\\n`) present. 
    detector: str, 'A' or 'B'
        The name of the detector in the file. Default is `B`. Note that only that only 
        'A' or 'B' is an acceptable input and not the complete detector name such as 
        'LC Chromatogram(Detector B-Ch1)'.
    delimiter: str 
        The delimeter used in the file. If  tab-delimited, use `\\t`.
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
    TypeError :
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
    chrom = file.split(
                    f'[LC Chromatogram(Detector {detector.upper()}-Ch'
                    )[1]
    if '[' in chrom:
        chrom = chrom.split('[')[0]
    
    chrom = '\n'.join(chrom.split('\n')[1:-2])
    # Read into as a dataframe.
    df = pd.read_csv(StringIO(chrom), delimiter=delimiter, skiprows=6)

    # Rename the columns
    df.columns = ['time_min', 'intensity_mV']
    df['detector'] = detector

    # Dropnas
    df.dropna(inplace=True)
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

def scrape_peak_table(file, detector='B', delimiter=','):
    """
    Scrapes the Shimadzu-generated peak table output from the ASCII output.

    Parameters
    ----------
    file : str 
        The contents of the file as a string with newline characters (`\\n`) present. 
    detector: str, 'A' or 'B'
        The name of the detector in the file. Default is `B`. Note that only that only 
        'A' or 'B' is an acceptable input and not the complete detector name such as 
        'Peak Table(Detector B)'. 
    delimiter: str 
        The delimeter used in the file. If  tab-delimited, use `\\t`.

    Returns
    -------
    peaks : pandas DataFrame
        A tidy pandas DataFrame with the identified peaks, times, areas, and 
        heights.

    Notes
    -----
    This function assumes that the detector name follows the convention 
    `[Peak Table(Detector A/B]` where `A/B` is the detector label 
    and `X` is the channel number. 

    Raises
    ------
    TypeError :
        Raised if file, detctor, or delimiter is not of type `str`.
    ValueError:
        Raised if `[LC Chromatogram(Detector`, `\\n`, or delimiter is not present in file.
        Also raised if `detector.upper()` is not `A` or `B`
    """
    # Do type checks. 
    for arg in (file, detector, delimiter):
        if type(arg) is not str:
            raise TypeError(f'Type of `{arg}` must be `str`. Type {type(arg)} was provided.')
    for st in ['[Peak Table(Detector', '\n', delimiter]:
        if st not in file:
            raise ValueError(f'Pattern {st} not present in file.')
    if (detector.upper() != 'A') & (detector.upper() != 'B'):
        raise ValueError(f'Detector must be `A` or `B`. String of `{detector.upper()}` was provided.')

    # Parse and split the file to get the chromatogram
    peaks = file.split(
                    f'[Peak Table(Detector {detector.upper()}'
                    )[1]
    if '[' in peaks:
        peaks = peaks.split('[')[0]
    
    peaks = '\n'.join(peaks.split('\n')[1:])
    
    # Read into as a dataframe.
    df = pd.read_csv(StringIO(peaks), delimiter=delimiter, skiprows=1)

    # Rename the columns
    df = df[['Peak#', 'R.Time', 'I.Time', 'F.Time', 'Area', 'Height']]
    df.rename(columns={'Peak#':'peak_idx', 'R.Time':'retention_time', 
                       'I.Time': 'arrival_time', 'F.Time':'departure_time',
                       'Area':'area', 'Height':'height'}, inplace=True)
    df['detector'] = detector

    # Dropnas
    df.dropna(inplace=True)

    # Determine if the metadata should be scraped and returned
    return df 

def convert(file_path, detector='B', delimiter=',', peak_table=False, 
            output_dir=None,  save_prefix=None, save_suffix=None, 
            verbose=True):
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
    peak_table: bool
        If True, the peak table is also parsed and is saved in the directory 
        with the extra suffix `_peaks`
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
        If True a progress bar will print if there's more than 5 files to 
        process.
    """

    # Determine the size of the  
    if type(file_path) is not list:
        file_path = [file_path]

    # Determine if there should be a verbose output
    if (verbose is True) & (len(file_path) >= 5):
        iterator = enumerate(tqdm.tqdm(file_path, desc='Converting ASCII output.'))
    else:
        iterator = enumerate(file_path)
    
    #TODO: Make sure this works on a windows system
    if output_dir is None:
        output_path = '/'.join(file_path[0].split('/')[:-1])
        if os.path.isdir(f'{output_path}/converted'):
            shutil.rmtree(f'{output_path}/converted')
        if os.path.isdir(f'{output_path}/converted') == False:
            os.mkdir(f'{output_path}/converted')
        output_path += '/converted'
    else:
        output_path = output_dir
    for _, f in iterator:
        with open(f, 'r') as file:
            raw_file = file.read()
            # Get the file metadata 
            file_metadata = scrape_metadata(raw_file, delimiter=delimiter)

            # Get chromatogram
            if type(detector) == str:
                detector = [detector]
            chroms = []
            peaks = []
            chrom_metadata = {}
            chrom_file_metadata = []
            for d in detector:

                # Parse the chromatogram
                _chrom, _chrom_metadata = scrape_chromatogram(raw_file, 
                                                            detector=d, 
                                                            delimiter=delimiter)    
                if peak_table:
                    _peaks = scrape_peak_table(raw_file,
                                               detector=d,
                                               delimiter=delimiter)
                    peaks.append(_peaks)
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
                if peak_table:
                    peaks = peaks[0]
            else:
                chrom = pd.concat(chrom, sort=False)
                if peak_table:
                    peaks = pd.concat(peaks, sort=False)
            if type(detector) == str:
                detector = [detector]

            # Assemble the comment file 
            header = f"""#
# {file_metadata['Sample Name']}
# ------------------------------
# Acquired: {file_metadata['Acquired']}
# Converted: {datetime.now().strftime('%m/%d/%Y %I:%m:%S %p')}
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
            if peak_table:
                peak_name = name + '_peak_table.csv'
                peak_save_name = f'{output_path}/{peak_name}'
            name += '_chromatogram.csv'
            save_name = f'{output_path}/{name}'

            if os.path.isfile(save_name):
                exists = True
                iter = 0
                while exists:
                    new_name = f'{save_name.split(".csv")[0]}_{iter}.csv'
                    if peak_table:
                        new_peak_name = f'{peak_save_name.split(".csv")[0]}_{iter}.csv'
                    if os.path.isfile(new_name):
                        iter += 1
                    else:
                        exists = False
                save_name = new_name
                peak_save_name = new_peak_name
            with open(save_name, 'a') as save_file:
                save_file.write(header)
                chrom.to_csv(save_file, index=False) 
            if peak_table:
                with open(peak_save_name, 'a') as save_file:
                    save_file.write(header)
                    peaks.to_csv(save_file, index=False) 

    print(f'Converted file(s) saved to `{output_path}`')

class Chromatogram(object):
    """
    Base class for dealing with HPLC chromatograms
    """
    def __init__(self, file=None, time_window=None,
                    cols={'time':'time_min', 'intensity':'intensity_mV'},
                    csv_comment='#'):
        """
        Instantiates a chromatogram object on which peak detection and quantification
        is performed.

        Parameters
        ----------
        file: str or pandas DataFrame, optional
            The path to the csv file of the chromatogram to analyze or 
            the pandas DataFrame of the chromatogram. If None, a pandas DataFrame 
            of the chromatogram must be passed.
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
        if file is None:
            raise RuntimeError(f'File path or dataframe must be provided')
        if (type(file) is not str) & (type(file) is not pd.core.frame.DataFrame):
            raise RuntimeError(f'Argument must be either a filepath or pandas DataFrame. Argument is of type {type(file)}')
        if (time_window is not None):
            if type(time_window) != list:
                raise TypeError(f'`time_window` must be of type `list`. Type {type(time_window)} was proivided')
            if len(time_window) != 2:
                raise ValueError(f'`time_window` must be of length 2 (corresponding to start and end points). Provided list is of length {len(time_window)}.')

        # Assign class variables 
        self.time_col = cols['time']
        self.int_col = cols['intensity']

        # Load the chromatogram and necessary components to self. 
        if type(file) is str:
            dataframe = pd.read_csv(file, comment='#')
        else:
            dataframe = file 
        self.df = dataframe

        # Prune to time window
        if time_window is not None:
            self.crop(time_window)
        else: 
            self.df = dataframe

         # Correct for a negative baseline 
        df = self.df
        min_int = df[self.int_col].min() 
        intensity = df[self.int_col] - min_int

        # Blank out vars that are used elsewhere
        self.window_df = None
        self.window_props = None
        self.peaks = None
        self.peak_df = None

    def crop(self, time_window=None, return_df=False):
        """
        Restricts the time dimension of the DataFrame

        Parameters
        ----------
        time_window : list [start, end], optional
            The retention time window of the chromatogram to consider for analysis.
            If None, the entire time range of the chromatogram will be considered.
        return_df : bool
            If `True`, the cropped DataFrame is 

        Returns
        -------
        cropped_df : pandas DataFrame
            If `return_df = True`, then the cropped dataframe is returned.
        """
        if type(time_window) != list:
                raise TypeError(f'`time_window` must be of type `list`. Type {type(time_window)} was proivided')
        if len(time_window) != 2:
                raise ValueError(f'`time_window` must be of length 2 (corresponding to start and end points). Provided list is of length {len(time_window)}.')
        self.df = self.df[(self.df[self.time_col] >= time_window[0]) & 
                          (self.df[self.time_col] <= time_window[1])]
        if return_df:
            return self.df

    def _assign_peak_windows(self, prominence, rel_height, buffer):
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
        for param, param_name, param_type in zip([prominence, rel_height, buffer], 
                                     ['prominence', 'rel_height',  'buffer'],
                                     [float, float, int]):
            if type(param) is not param_type:
                raise TypeError(f'Parameter {param_name} must be of type `{param_type}`. Type `{type(param)}` was supplied.') 
        if (prominence < 0) | (prominence > 1):
            raise ValueError(f'Parameter `prominence` must be [0, 1].')
        if (rel_height < 0) | (rel_height > 1):  
            raise ValueError(f'Parameter `rel_height` must be [0, 1].')
        if (buffer < 0):
            raise ValueError('Parameter `buffer` cannot be less than 0.')

        # Correct for a negative baseline 
        df = self.df
        intensity = self.df[self.int_col].values
        norm_int = (intensity - intensity.min()) / (intensity.max() - intensity.min())

        # Identify the peaks and get the widths and baselines
        peaks, _ = scipy.signal.find_peaks(norm_int, prominence=prominence)
        self.peaks_inds = peaks
        out = scipy.signal.peak_widths(intensity, peaks, 
                                       rel_height=rel_height)
        _, heights, left, right = out
        widths, _, _, _ = scipy.signal.peak_widths(intensity, peaks, 
                                       rel_height=0.5)

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
            window_df.loc[window_df['time_idx'].isin(r), 
                                    'window_idx'] = int(i + 1)
            window_df.loc[window_df['time_idx'].isin(r), 
                                    'baseline'] = baselines[i]
        window_df.dropna(inplace=True) 

        # Convert this to a dictionary for easy parsing
        window_dict = {}
        time_step = np.mean(np.diff(self.df[self.time_col].values))
        for g, d in window_df.groupby('window_idx'):
            _peaks = [p for p in peaks if p in d['time_idx'].values]
            peak_inds = [x for _p in _peaks for x in np.where(peaks == _p)[0]]
            _dict = {'time_range':d[self.time_col].values,
                     'intensity': d[self.int_col] - baselines[i],
                     'num_peaks': len(_peaks),
                     'amplitude': [d[d['time_idx']==p][self.int_col].values[0] - baselines[i] for p in _peaks],
                     'location' : [d[d['time_idx']==p][self.time_col].values[0] for p in _peaks],
                     'width' :    [widths[ind] * time_step for ind in peak_inds]
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

        Notes
        -----
        This function infers the parameters defining skew-norma distributions 
        for each peak in the chromatogram. The fitted distribution has the form 
            
        .. math:: 
            I = 2I_\text{max} \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)e^{-\frac{(t - r_t)^2}{2\sigma^2}}\left[1 + \text{erf}\frac{\alpha(t - r_t)}{\sqrt{2\sigma^2}}\right]

        where :math:`I_\text{max}` is the maximum intensity of the peak, 
        :math:`t` is the time, :math:`r_t` is the retention time, :math:`\sigma`
        is the scale parameter, and :math:`\alpha` is the skew parameter.

        """
        amp, loc, scale, alpha = params
        _x = alpha * (x - loc) / scale
        norm = np.sqrt(2 * np.pi * scale**2)**-1 * np.exp(-(x - loc)**2 / (2 * scale**2))
        cdf = 0.5 * (1 + scipy.special.erf(_x / np.sqrt(2))) 
        return amp * 2 * norm * cdf

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
            iterator = tqdm.tqdm(self.window_props.items(), desc='Fitting peak windows...')  
        else:
            iterator = self.window_props.items()

        peak_props = {}
        for k, v in iterator:
            window_dict = {}
            # Set up the initial guess
            p0 = [] 
            bounds = [[],  []] 
            for i in range(v['num_peaks']):
                p0.append(v['amplitude'][i])
                p0.append(v['location'][i]),
                p0.append(v['width'][i] / 2) # scale parameter
                p0.append(0) # Skew parameter, starts with assuming Gaussian

                # Set the bounds 
                bounds[0].append(0)
                bounds[0].append(v['time_range'].min())
                bounds[0].append(0)
                bounds[0].append(-np.inf)
                bounds[1].append(np.inf)
                bounds[1].append(v['time_range'].max())
                bounds[1].append(np.inf)
                bounds[1].append(np.inf)

            # Perform the inference
            try:
                popt, _ = scipy.optimize.curve_fit(self._fit_skewnorms, v['time_range'],
                                               v['intensity'], p0=p0, bounds=bounds,
                                               maxfev=int(1E4))

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
                                'alpha': p[3],
                                'area':self._compute_skewnorm(v['time_range'], *p).sum()}
                peak_props[k] = window_dict
            except RuntimeError:
                print('Warning: Parameters could not be inferred for one peak')
        self.peak_props = peak_props
        return peak_props

    def quantify(self, time_window=None, prominence=1E-3, rel_height=1.0, 
                 buffer=100, verbose=True):
        R"""
        Quantifies peaks present in the chromatogram

        Parameters
        ----------
        time_window: list [start, end], optional
            The retention time window of the chromatogram to consider for analysis.
            If None, the entire time range of the chromatogram will be considered.
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
        verbose : bool
            If True, a progress bar will be printed during the inference. 

        Returns
        -------
        peak_df : pandas DataFrame
            A dataframe containing information for each detected peak.


        Notes
        -----
        This function infers the parameters defining skew-norma distributions 
        for each peak in the chromatogram. The fitted distribution has the form 
            
        .. math:: 
            I = 2I_\text{max} \left(\frac{1}{\sqrt{2\pi\sigma^2}}\right)e^{-\frac{(t - r_t)^2}{2\sigma^2}}\left[1 + \text{erf}\frac{\alpha(t - r_t)}{\sqrt{2\sigma^2}}\right]

        where :math:`I_\text{max}` is the maximum intensity of the peak, 
        :math:`t` is the time, :math:`r_t` is the retention time, :math:`\sigma`
        is the scale parameter, and :math:`\alpha` is the skew parameter.

        """
        if time_window is not None:
            dataframe = self.df
            self.df = dataframe[(dataframe[self.time_col] >= time_window[0]) & 
                              (dataframe[self.time_col] <= time_window[1])].copy(deep=True) 
        # Assign the window bounds
        _ = self._assign_peak_windows(prominence, rel_height, buffer)

        # Infer the distributions for the peaks
        peak_props = self._estimate_peak_params(verbose)

        # Set up a dataframe of the peak properties
        peak_df = pd.DataFrame([])
        iter = 0 
        for _, peaks in peak_props.items():
            for _, params in peaks.items():
                _dict = {'retention_time': params['retention_time'],
                         'scale': params['std_dev'],
                         'skew': params['alpha'],
                         'amplitude': params['amplitude'],
                         'area': params['area'],
                         'peak_idx': iter + 1}     
                iter += 1
                peak_df = peak_df.append(_dict, ignore_index=True)
                peak_df['peak_idx'] = peak_df['peak_idx'].astype(int)
        self.peak_df = peak_df

        # Compute the mixture
        time = self.df[self.time_col].values
        out = np.zeros((len(time), len(peak_df)))
        iter = 0
        for _k , _v in self.peak_props.items():
            for k, v in _v.items():
                params = [v['amplitude'], v['retention_time'], 
                          v['std_dev'], v['alpha']]
                out[:, iter] = self._compute_skewnorm(time, *params)
                iter += 1
        self.mix_array = out
        return peak_df

    def show(self):
        """
        Displays the chromatogram with mapped peaks if available.
        """
        sns.set()

        # Set up the figure    
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlabel(self.time_col)
        ax.set_ylabel(self.int_col)

        # Plot the raw chromatogram
        ax.plot(self.df[self.time_col], self.df[self.int_col], 'k-', lw=2,
                label='raw chromatogram') 

        # Compute the skewnorm mix 
        if self.peak_df is not None:
            time = self.df[self.time_col].values
            # Plot the mix
            convolved = np.sum(self.mix_array, axis=1)
            ax.plot(time, convolved, 'r--', label='inferred mixture') 
            for i in range(len(self.peak_df)):
                ax.fill_between(time, self.mix_array[:, i], label=f'peak {i+1}', 
                                alpha=0.5)
        ax.legend(bbox_to_anchor=(1,1))
        fig.patch.set_facecolor((0, 0, 0, 0))
        return [fig, ax]


def batch_process(file_paths, time_window=None,  show_viz=False,
                  cols={'time':'time_min', 'intensity':'intensity_mV'},  
                  **kwargs):
    """
    Performs complete quantification of a set of HPLC data. Data must first 
    be converted to a tidy long-form CSV file by using `cremerlab.hplc.convert`

    Parameters
    ----------
    file_paths : list of str
        A list of the file paths.
    time_window : list, optional
        The upper and lower and upper time bounds to consider for the analysis.
        Default is `None`, meaning the whole chromatogram is considered.
    show_viz : bool 
        If `True`, the plot is printed to the screen. If `False`, a plot is 
        generated and returned, just not shown.
    cols: dict, keys of 'time', and 'intensity', optional
        A dictionary of the retention time and intensity measurements of the 
        chromatogram. Default is `{'time':'time_min', 'intensity':'intensity_mV'}`.
    kwargs: dict, **kwargs
        **kwargs for the peak quantification function `cremerlab.hplc.Chromatogram.quantify`

    Returns 
    --------
    chrom_df : pandas DataFrame
        A pandas DataFrame  of all of the chromatograms, indexed by file name
    peak_df : pandas DataFrame
        A pandas DataFrame of all identified peaks, indexed by file name 
    fig : matplotlib.figure.Figure
        Matplotlib figure object for the chromatograms
    ax :  matplotlib AxesSubplot
        The axes of the matplotlib figure
    """

    # Instantiate storage lists
    chrom_dfs, peak_dfs, mixes = [], [], []

    # Perform the processing for each file
    for i, f in enumerate(tqdm.tqdm(file_paths, desc='Processing files...')):
        # Generate the sample id
        if '/' in f:
            file_name = f.split('/')[-1]
        else:
            file_name = f

        # Check for common file name extension
        for pat in ['.csv', '.txt']:
            if pat in file_name:
                file_name = file_name.split(pat)[0]
                continue

        # Parse teh chromatogram and quantify the peaks
        chrom = Chromatogram(f, cols=cols, time_window=time_window)
        peaks = chrom.quantify(verbose=False, **kwargs)

        # Set up the dataframes for chromatograms and peaks
        _df = chrom.df
        _df['sample'] = file_name
        peaks['sample'] = file_name
        peak_dfs.append(peaks)
        chrom_dfs.append(chrom.df)
        mixes.append(chrom.mix_array)

    # Concateante the dataframe
    chrom_df = pd.concat(chrom_dfs, sort=False)
    peak_df = pd.concat(peak_dfs, sort=False) 

    # Determine the size of the figure
    num_traces = len(chrom_df['sample'].unique())
    num_cols = int(2)
    num_rows = int(np.ceil(num_traces / num_cols))
    unused_axes = (num_cols * num_rows) - num_traces

    # Instantiate the figure
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 2 * num_rows))

    ax = ax.ravel()

    for a in ax:
        a.xaxis.set_tick_params(labelsize=6)
        a.yaxis.set_tick_params(labelsize=6)
        a.set_ylabel(cols['intensity'], fontsize=6)
        a.set_xlabel(cols['time'], fontsize=6)
    for i in range(unused_axes):
        ax[-(i + 1)].axis('off')

    # Assign samples to axes.
    mapper = {g: i for i, g in enumerate(chrom_df['sample'].unique())} 

    # Plot the chromatogram
    for g, d in chrom_df.groupby(['sample']): 
        ax[mapper[g]].plot(d[cols['time']], d[cols['intensity']], 'k-', lw=0.5,
                           label='raw chromatogram')
        ax[mapper[g]].set_title(' '.join(g.split('_')), fontsize=6)

    # Plot the mapped peaks
    for g, d in peak_df.groupby(['sample']):
        mix = mixes[mapper[g]] 
        convolved = np.sum(mix, axis=1)
        for i in range(len(d)): 
            _m = np.array(mix[:, i])
            time = np.linspace(time_window[0], time_window[1], len(_m))
            ax[mapper[g]].fill_between(time, 0, _m, alpha=0.5, label=f'peak {i + 1}')

        time = np.linspace(time_window[0], time_window[1], len(convolved))
        ax[mapper[g]].plot(time, convolved, '--', color='red', lw=0.5, 
                            label=f'inferred mixture')
    plt.tight_layout()
    fig.patch.set_facecolor((0, 0, 0, 0))
    ax[0].legend(fontsize=6)
    if show_viz == False:
        plt.close()
    return [chrom_df, peak_df, [fig, ax]]

