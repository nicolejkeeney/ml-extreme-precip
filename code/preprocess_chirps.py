"""Preprocess CHIRPS precip data for use in CNN

1) Clip the data to a geometry of interest. 
2) Compute spatial average. 
3) Compute 95th percentile. Assign classes: 0 (below 95th perc) or 1 (above 95th perc)

Author:
    Nicole Keeney 
    
Version: 
    10-30-2023 

References: 
    Code builds off code from Davenport and Diffenbaugh, 2021: https://github.com/fdavenport/GRL2021/blob/main/notebooks/0a_read_reanalysis.ipynb
    
"""

import xarray as xr
import pandas as pd
from glob import glob
import os
import time

# Import helper functions
from utils.preprocessing_utils import get_us_states_geom, clip_to_geom
import utils.parameters as param
from utils.misc_utils import check_and_create_dir

# ------------ SET GLOBAL VARIABLES ------------

CHIRPS_DATA_DIR = "../data/chirps_precip/" # Raw CHIRPS data dir 
OUTPUT_DATA_DIR = "../data/input_data_preprocessed/labels/" # Where to store preprocessed data 
SHP_DIR = "../data/cb_2018_us_state_5m/" # US states shapefile
GEOM_NAME = "Arizona"

def chirps_5x5(chirps_data_dir=CHIRPS_DATA_DIR, output_data_dir=OUTPUT_DATA_DIR, output_filename="chirps_5x5"): 
    """Get CHIRPS data on a 5x5 grid. 
    Compute 95% quantile and assign precip classes.
    Save output to netcdf file.
    
    Parameters 
    ----------
    chirps_data_dir: str 
        Path to raw CHIRPS data 
    output_data_dir: str 
        Path to directory for storing output file
    output_filename: str, optional 
        Name to give output file. Do not include extension. 
        Default to "chirps_5x5"
    
    Returns 
    -------
    ds_training, ds_validation, ds_testing: xr.Dataset 
        Datasets split by time slice for each dataset type as set in param module 
        Datasets have the following data variables:
            (1) precip: Original data, on coarsened grid 
            (2) p95: 95th percentile for each gridcell
            (3) precip_class: Assigned value of 0,1 depending on exceedence of 95th percentile  

    """

    # Read in data
    var = "precip"
    filepaths_wildcard = chirps_data_dir +"*chirps*.days_p25.nc"
    filepaths_all = glob(filepaths_wildcard)
    ds = xr.open_mfdataset(filepaths_all).sel(time=param.time_period)

    # Coarsen data to smaller grid. Compute mean to downsample
    ds = ds.chunk(dict(time=-1))
    ds_coarsened = ds.coarsen(latitude=20,longitude=20).mean()

    # Compute 95th perc across time dimension for each gridcell 
    # Add as data variable to Dataset
    ds_coarsened["p95"] = ds_coarsened[var].quantile(0.95, dim="time")

    # Assign classes based on exceedance of 95th percentile
    ds_coarsened["precip_class"] = xr.where(ds_coarsened[var] > ds_coarsened["p95"], 1, 0)
    ds_coarsened["precip_class"].attrs = {
        "classes": "Class 0: precipitation below threshold \nClass 1: precipitation exeeds threshold",
    }

    # Split into training, validation, and testing
    ds_training = ds_coarsened.sel(time = slice(param.training_time_start, param.training_time_end))
    ds_validation = ds_coarsened.sel(time = slice(param.validation_time_start, param.validation_time_end))
    ds_testing = ds_coarsened.sel(time = slice(param.testing_time_start, param.testing_time_end))

    # Save to netcdf 
    filepath = "{0}{1}".format(output_data_dir, output_filename)
    for ds_i, ds_name in zip([ds_training, ds_validation, ds_testing], ["training","validation","testing"]):
        path_i = "{0}_{1}.nc".format(filepath, ds_name)
        ds_i.to_netcdf(path_i)
        print("netcdf for {0} dataset saved to {1}".format(ds_name, path_i))
    
    return ds_training, ds_validation, ds_testing

def chirps_by_state(geom_name=GEOM_NAME, chirps_data_dir=CHIRPS_DATA_DIR, shp_dir=SHP_DIR, output_data_dir=OUTPUT_DATA_DIR):
    """Preprocess CHIRPS data by US state. 
    Save as csv file split by training, validation, and testing as set in param module. 
    
    Parameters 
    ----------
    geom_name: str
        Name of US state. 
        Must follow naming conventions in the shapefile! 
        State name must be capitalized i.e. "Missouri" not "missouri"
    chirps_data_dir: str 
        Path to raw CHIRPS data
    shp_dir: str 
        Path to shapefile of US state boundaries 
    output_data_dir: str 
        Path to directory for storing output file

    Returns 
    -------
    output_df: pd.DataFrame 
    
    """
    # Get geometry
    geom = get_us_states_geom(state=geom_name, shp_path=shp_dir)

    # Make directory for saving preprocessed data if it doesn't already exist
    # Replace spaces with underscores in state name 
    preprocessed_data_dir = output_data_dir + geom_name.replace(" ", "_") + "/"
    check_and_create_dir(preprocessed_data_dir)

    # Read in data
    var = "precip"
    filepaths_wildcard = chirps_data_dir + "*chirps*.days_p25.nc"
    filepaths_all = glob(filepaths_wildcard)
    ds = xr.open_mfdataset(filepaths_all).sel(time=param.time_period)

    # Clip to geometry
    ds = clip_to_geom(ds, geom, lon_name="longitude", lat_name="latitude")

    # ## Plot the data (in notebook only)
    # # Sanity check to make sure the clipping functioned as expected
    # import geopandas as gpd
    # import hvplot.xarray
    # import hvplot.pandas

    # # US states
    # not_CONUS = ["Alaska","Hawaii","Commonwealth of the Northern Mariana Islands", "Guam", "American Samoa", "Puerto Rico","United States Virgin Islands"]
    # us_states = gpd.read_file(shp_dir)
    # conus = us_states[~us_states["NAME"].isin(not_CONUS)]
    # boundary_pl = conus.hvplot(color=None)

    # # Geospatial data
    # geo_pl = ds.isel(time=0).hvplot(x="longitude",y="latitude", cmap="viridis_r")
    # geo_pl*boundary_pl

    # Average over entire region
    ds_mean = ds.mean(dim=["latitude", "longitude"])

    # Rename variable
    ds_mean = ds_mean.rename({var: var + "_mean"})

    # Read data into memory
    ds_mean = ds_mean.compute()

    # Compute 95th percentile precip
    perc_95 = ds_mean[var + "_mean"].quantile(0.95).item()
    print("95th percentile precip over {0}: {1}".format(GEOM_NAME, perc_95))

    # Assign classes based on exceedance of 95th percentile
    extremes_var = "precip_class"
    ds_mean[extremes_var] = xr.where(ds_mean[var + "_mean"] > perc_95, 1, 0)
    ds_mean[extremes_var].attrs = {
        "description": "95th percentile precipitation",
        "classes": "Class 0: precipitation below threshold \nClass 1: precipitation exeeds threshold",
        "95th percentile": "{} mm/day".format(round(perc_95, 3)),
    }

    # Convert to dataframe
    output_df = ds_mean.to_dataframe()

    # Split into training-validation-testing
    training = output_df.loc[param.training_time_start : param.training_time_end]
    validation = output_df.loc[param.validation_time_start : param.validation_time_end]
    testing = output_df.loc[param.testing_time_start : param.testing_time_end]

    # Output as csv
    training.to_csv(preprocessed_data_dir + "training_labels.csv")
    validation.to_csv(preprocessed_data_dir + "validation_labels.csv")
    testing.to_csv(preprocessed_data_dir + "testing_labels.csv")

    return output_df 

if __name__ == "__main__":
    chirps_by_state()
