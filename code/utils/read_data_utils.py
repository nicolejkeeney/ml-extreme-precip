""" Module for reading and formatting model input data 

Author:
    Nicole Keeney 
    
Version: 
    11-06-2023

"""

import numpy as np 
import xarray as xr 
import pandas as pd 


def get_input_data(data_dir, settings): 
    """Retrieve and format input data to model 

    Parameters 
    ---------
    data_dir: str 
        Path to data directory 
    settings: dict 
        Model settings 
    
    Returns 
    -------
    x_train_ds: xr.Dataset
        Featurs for training data as xarray object 
    y_train_df: pd.DataFrame 
        Labels for training data + any additional data from the original csv 
    x_val_ds: xr.Dataset
        Featurs for validation data as xarray object  
    y_val_df: pd.DataFrame 
        Labels for validation data + any additional data from the original csv 
    x_test_ds: xr.Dataset
        Featurs for testing data as xarray object 
    y_test_df: pd.DataFrame 
        Labels for testing data + any additional data from the original csv 

    """

    # Read labels csv as pandas DataFrame object
    # Read features netcdf as xarray object
    x_train_ds = xr.open_dataset(
        data_dir + "features/" + settings["features_geom"] + "/" + settings["features_geom"] + "_training_features.nc"
    )
    y_train_df = pd.read_csv(
        data_dir + "labels/" + settings["labels_geom"] + "/training_labels.csv",
        index_col=False,
    )
    x_val_ds = xr.open_dataset(
        data_dir + "features/" + settings["features_geom"] + "/" + settings["features_geom"] + "_validation_features.nc"
    )
    y_val_df = pd.read_csv(
        data_dir + "labels/" + settings["labels_geom"] + "/validation_labels.csv",
        index_col=False,
    )
    x_test_ds = xr.open_dataset(
        data_dir + "features/" + settings["features_geom"] + "/" + settings["features_geom"] + "_testing_features.nc"
    )
    y_test_df = pd.read_csv(
        data_dir + "labels/" + settings["labels_geom"] + "/testing_labels.csv", index_col=False
    )

    return (
        x_train_ds, y_train_df, 
        x_val_ds, y_val_df, 
        x_test_ds, y_test_df
    ) 


def format_input_data(x_train_ds, y_train_df, x_val_ds, y_val_df, x_test_ds, y_test_df):
    """Convert raw input data into numpy arrays. Do one hot encoding on labels 
    
    Parameters 
    ----------
    x_train_ds: xr.Dataset
        Featurs for training data as xarray object 
    y_train_df: pd.DataFrame 
        Labels for training data + any additional data from the original csv 
    x_val_ds: xr.Dataset
        Featurs for validation data as xarray object  
    y_val_df: pd.DataFrame 
        Labels for validation data + any additional data from the original csv 
    x_test_ds: xr.Dataset
        Featurs for testing data as xarray object 
    y_test_df: pd.DataFrame 
        Labels for testing data + any additional data from the original csv 

    Returns 
    --------
    x_train: np.array 
        Features for training data (4D array)
        Shape of array is (time, lat, lon, variable)
    y_train: np.array 
        Labels for training data (1D array)
    y_train_onehot: np.array 
        Labels for training data, one hot encoded (2D array)
    x_val: np.array
        Features for validation data
    y_val: np.array
        Labels for validation data 
    y_val_onehot: np.array
        One hot encoded labels for validation data 
    x_test: np.array
        Features for testing data 
    y_test: np.array
        Labels for testing data
    y_test_onehot: np.array
        One hot encoded labels for testing data 
    
    """

    labels = "precip_classes"

    # Convert data to numpy arrays
    x_train = x_train_ds.to_array().transpose("time", "lat", "lon", "variable").values
    y_train = y_train_df[labels].values

    x_val = x_val_ds.to_array().transpose("time", "lat", "lon", "variable").values
    y_val = y_val_df[labels].values

    x_test = x_test_ds.to_array().transpose("time", "lat", "lon", "variable").values
    y_test = y_test_df[labels].values

    # Do one hot encoding on labels
    y_train_onehot = onehot(y_train)
    y_val_onehot = onehot(y_val)
    y_test_onehot = onehot(y_test)

    return (
        x_train, y_train, y_train_onehot,  
        x_val, y_val, y_val_onehot, 
        x_test, y_test, y_test_onehot
    )

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
    y = np.zeros((x.size, x.max() + 1))  ## create new array
    y[np.arange(x.size), x] = 1

    return y