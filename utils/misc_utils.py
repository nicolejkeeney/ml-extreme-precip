# Miscellaneous helper functions

import math

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