"""Preprocess NCEP-NCAR R1 reanalysis data for use in CNN

1) Clip the data to a geometry of interest. 
2) [geopotential heights only] detrend the data 
3) Compute spatial average. 
4) Compute daily standardized anomalies. 

Author:
    Nicole Keeney 
    
Version: 
    10-30-2023 

References: 
    Code builds off code from Davenport and Diffenbaugh, 2021: https://github.com/fdavenport/GRL2021/blob/main/notebooks/0a_read_reanalysis.ipynb
    
"""

import xarray as xr
import numpy as np
from glob import glob
from datetime import datetime
import geopandas as gpd
import os

# Import helper functions
from utils.preprocessing_utils import (
    get_custom_geom
    convert_lon_360_to_180,
    clip_to_geom,
    calc_anomalies,
)
from utils.misc_utils import (
    format_nbytes, 
    check_and_create_dir
)
import utils.parameters as param

## GLOBAL VARIABLES 

DATA_DIR = "../data/"
GEOM_NAME = "CONUS"

def main(geom_name=GEOM_NAME, data_dir=DATA_DIR):  

    # Get geometry
    geom = get_custom_geom(geom_name=geom_name)

    # Make directory for saving preprocessed data if it doesn't already exist
    preprocessed_data_dir = data_dir + "input_data_preprocessed/labels/" + geom_name +"/"
    check_and_create_dir(preprocessed_data_dir)

    ## -------- Preprocess sea level pressure data --------

    print("Preprocessing sea level pressure data...")

    # Open dataset
    var = "slp"  # Variable name
    filepaths_wildcard = data_dir + "{0}_daily_means/{1}*.nc".format(var, var)
    filepaths_all = glob(filepaths_wildcard)
    ds = xr.open_mfdataset(filepaths_all).sel(time=param.time_period)
    global_attrs = ds.attrs
    ds = ds.drop_dims("nbnds")

    # Convert lon range from 0:360 to -180:180
    ds = convert_lon_360_to_180(ds)

    # Clip to geometry
    ds = clip_to_geom(ds, geom)

    # Calculate daily standardized anomalies
    ds = calc_anomalies(ds, var)

    # Format the output data
    slp_output_da = ds[var + "_anom"]
    slp_output_da.attrs = {
        "long_name": "mean daily sea level pressure anomalies",
        "units": "Pa",
    }

    ## --- Plot the data (in a python notebook only)
    # import hvplot.xarray
    # import hvplot.pandas

    # # US states
    # shp_path = DATA_DIR+"cb_2018_us_state_5m/"
    # not_CONUS = ["Alaska","Hawaii","Commonwealth of the Northern Mariana Islands", "Guam", "American Samoa", "Puerto Rico","United States Virgin Islands"]
    # us_states = gpd.read_file(shp_path)
    # conus = us_states[~us_states["NAME"].isin(not_CONUS)]
    # boundary_pl = conus.hvplot(color=None)

    # # Geospatial data
    # geo_pl = slp_output_da.hvplot(x="lon",y="lat", cmap="coolwarm")
    # geo_pl*boundary_pl

    ## -------- Preprocess geopotential height data --------

    print("Preprocessing geopotential height data...")

    # Open dataset
    var = "hgt"
    filepaths_wildcard = data_dir + "{0}_daily_means/{1}*.nc".format(var, var)
    filepaths_all = glob(filepaths_wildcard)
    ds = xr.open_mfdataset(filepaths_all).sel(time=param.time_period)
    global_attrs = ds.attrs

    # Clean it up a bit
    level = 500
    ds = ds.sel(time=param.time_period)
    ds = ds.drop_dims("nbnds")
    ds = ds.sel(level=level).drop("level")

    # Convert lon range from 0:360 to -180:180
    ds = convert_lon_360_to_180(ds)

    # Clip to geometry
    ds = clip_to_geom(ds, geom)

    # Calculate annual domain average 500-hPa GPH to remove seasonal variability
    domain_mean_df = ds[var].groupby("time.year").mean(dim="time").to_dataframe(name=var)

    # Calculate linear trend in 500-hPa GPH
    trend = np.polyfit(
        domain_mean_df.index.get_level_values("year"), domain_mean_df[var], 1
    )
    print("Slope of trend:", trend[0], "m per year")

    # Calculate detrended hgt
    ds["change"] = (ds.time.dt.year - int(param.time_start[:4])) * trend[0]
    ds[var + "_detrended"] = ds[var] - ds["change"]
    ds = ds.drop_vars("change")

    # Calculate daily standardized anomalies
    ds = calc_anomalies(ds, var + "_detrended")

    # Format the output data
    hgt_output_da = ds[var + "_detrended_anom"]
    hgt_output_da.attrs = {
        "long_name": "mean detrended daily geopotential height anomalies",
        "units": "m",
        "level": level,
    }

    # # Plot the data (in a python notebook only)
    # geo_pl = hgt_output_da.hvplot(x="lon",y="lat", cmap="coolwarm")
    # geo_pl*boundary_pl

    ## -------- Combine datasets for both variables and write to netcdf --------

    print("Combining both variables and writing data to netcdfs...")

    # Merge DataArrays
    output_ds = xr.merge([hgt_output_da, slp_output_da])

    # Add descriptive attributes
    output_ds.attrs = global_attrs
    output_ds.attrs["title"] = (
        global_attrs["title"] + " modified to produce daily anomalies"
    )
    output_ds.attrs["history"] = (
        global_attrs["history"]
        + "\nDaily detrended anomalies produced "
        + datetime.today().strftime("%Y/%m/%d")
    )

    # Print size of dataset
    nbytes = format_nbytes(output_ds.nbytes)
    print("Size of output dataset: {0}".format(nbytes))

    # Split into training-validation-testing
    training = output_ds.sel(time=param.training_period)
    validation = output_ds.sel(time=param.validation_period)
    testing = output_ds.sel(time=param.testing_period)

    # Output to netcdf
    training.to_netcdf(preprocessed_data_dir + "training_features.nc")
    validation.to_netcdf(preprocessed_data_dir + "validation_features.nc")
    testing.to_netcdf(preprocessed_data_dir + "testing_features.nc")

if __name__ == "__main__":
    main()