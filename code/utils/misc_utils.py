"""Miscellanous helper functions 

Author: 
    Nicole Keeney 

Version: 
    11/8/2023
    
"""

import math
import json
import os 

def get_season_str(mon): 
    """Get string ID for each season using month int 
    
    Parameters 
    ----------
    mon: int 
        Integer corresponding to the month in the year (1-12)
    
    Returns 
    -------
    season: str 
        One of "DJF", "MAM", "JJA", "SON"
    """
    
    if mon in [12,1,2]: 
        season = "DJF"
    elif mon in [3,4,5]: 
        season = "MAM"
    elif mon in [6,7,8]: 
        season = "JJA"
    elif mon in [9,10,11]: 
        season = "SON"
    return season 

def add_to_settings_json(m_id, settings, settings_path="model_settings.json"): 
    """Add settings dictionary to json file 
    
    Parameters
    ----------
    m_id: str
        String identifier for model 
    settings: dict 
        Dictionary of settings 
    settings_path: str, optional 
        Path to settings file. Default to "model_settings.json".
        If the file doesn't exist, it will be created 
    
    Returns 
    -------
    None 

    """

    try: 
        # If the file exists, append to existing dict 
        f = open(settings_path)
        settings_all = json.load(f)
        settings_all[m_id] = settings
    except ValueError or FileNotFoundError as err: 
        # JSONDecode error is a type of ValueError 
        # Error is raised if file exists, but it is completely empty
        # Need to add an empty dict to the file 
        with open(settings_path, "w") as outfile:
            json.dump({}, outfile)
        settings_all = {m_id:settings}
    with open(settings_path, "w") as outfile:
        json.dump(settings_all, outfile)
        print("Settings for model {0} saved to {1}".format(m_id, settings_path))


def check_and_create_dir(dir): 
    """Check if directory exists. Create it if it does not. 
    
    Parameters 
    ----------
    dir: str or list of strs 
        Directory or list of directories 
    
    Returns 
    -------
    None 

    """

    if type(dir) != list: 
        dir = [dir] # Convert to list so that you can loop through 

    for dir_i in dir: 
        if not os.path.exists(dir_i):
            os.makedirs(dir_i)
            print("Created directory: {0}".format(dir_i))
    
    return None
    

def get_model_settings(model_id, settings_path="model_settings.json", print_to_console=True): 
    """Retrieve model settings for a given model ID 

    Parameters
    ----------
    model_id: str 
        Name of model. Must correspond to key in json file  
    settings_path: str, optional
        Path to model settings json file 
    print_to_console: boolean, optional
        Print settings to console? Default to True
    
    Returns 
    -------
    dict
        Dictionary of settings
    """

    f = open(settings_path)
    settings = json.load(f)[model_id]
    if print_to_console: 
        print("model id: {0}".format(model_id))
        print("model settings: {0}".format(json.dumps(settings, indent=4)))
    return settings



def format_nbytes(nbytes):
    """Format bytes into a readable string

    Parameters
    -----------
    nbytes: int
        Size in bytes

    Returns
    -------
    str

    References
    -----------
    Function credit to: https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python

    """
    if nbytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(nbytes, 1024)))
    p = math.pow(1024, i)
    s = round(nbytes / p, 2)
    return "%s %s" % (s, size_name[i])
