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

DATA_DIR = "../data/"
GEOM_NAME = "Arizona"

def main(geom_name=GEOM_NAME, data_dir=DATA_DIR):

    # Get geometry
    geom = get_us_states_geom(state=geom_name, shp_path=data_dir + "cb_2018_us_state_5m/")

    # Make directory for saving preprocessed data if it doesn't already exist
    # Replace spaces with underscores in state name 
    preprocessed_data_dir = data_dir + "input_data_preprocessed/labels/" + geom_name.replace(" ", "_") + "/"
    check_and_create_dir(preprocessed_data_dir)

    # Read in data
    var = "precip"
    filepaths_wildcard = data_dir + "chirps_precip/*chirps*.days_p25.nc"
    filepaths_all = glob(filepaths_wildcard)
    ds = xr.open_mfdataset(filepaths_all).sel(time=param.time_period)
    global_attrs = ds.attrs
    var_attrs = ds[var].attrs

    # Clip to geometry
    ds = clip_to_geom(ds, geom, lon_name="longitude", lat_name="latitude")

    # ## Plot the data (in notebook only)
    # # Sanity check to make sure the clipping functioned as expected
    # import geopandas as gpd
    # import hvplot.xarray
    # import hvplot.pandas

    # # US states
    # shp_path = DATA_DIR+"cb_2018_us_state_5m/"
    # not_CONUS = ["Alaska","Hawaii","Commonwealth of the Northern Mariana Islands", "Guam", "American Samoa", "Puerto Rico","United States Virgin Islands"]
    # us_states = gpd.read_file(shp_path)
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
    extremes_var = "precip_classes"
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

if __name__ == "__main__":
    main()
