import pandas as pd 
from io import StringIO
import tqdm
import os 
from datetime import datetime

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



