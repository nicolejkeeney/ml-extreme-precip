# Helper functions for convolutional neural network model

import numpy as np 

def onehot(x):
    """Convert x into onehot format

    Parameters 
    -----------
    x: np.array 
        1D array 

    Returns 
    np.array
        2D array

    References 
    ----------
    https://github.com/fdavenport/GRL2021/blob/main/project_utils/utils.py
    
    """
    y = np.zeros((x.size, x.max()+1)) ## create new array
    y[np.arange(x.size),x] = 1
    
    return(y)