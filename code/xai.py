"""Perform XAI on trained model

Author: 
    Nicole Keeney 

Version: 
    12-07-2023 

"""

import tensorflow as tf
import xarray as xr
import numpy as np
import pandas as pd
import innvestigate

import matplotlib.pyplot as plt 
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from utils.xai_utils import kmeans_clusters_xr
from utils.read_data_utils import (
    get_input_data, 
    format_input_data
)
from utils.misc_utils import (
    check_and_create_dir, 
    get_model_settings, 
)
from utils.figs_utils import make_heatmap_by_col

# innvestigate requires reverting to tensorflow version 1 
tf.compat.v1.disable_eager_execution()

# ------------ SET GLOBAL VARIABLES ------------

MODEL_ID = "frances_California"
STATE = MODEL_ID.split("frances_")[1]

DATA_DIR = "../data/input_data_preprocessed/us_states/" 
MODEL_OUTPUT_DIR = "../model_output/us_states/"+MODEL_ID+"/"
FIGURES_DIR = "../figures/us_states/"+MODEL_ID+"/xai/" 

NUM_CLUSTERS = 3 # Number of clusters to use for kmeans algorithm


def main(model_id=MODEL_ID, data_dir=DATA_DIR, model_output_dir=MODEL_OUTPUT_DIR, figures_dir=FIGURES_DIR, num_clusters=NUM_CLUSTERS): 
    """
    XAI analysis on trained model 
    Computes normalized layerwise relevance propagation interpretations and generates heatmap of results 
    Performs k-means clustering algorithm and generates heatmap of results 
    
    Parameters 
    ----------
    model_id: str
        String ID corresponding to the model you wish to run the function for 
        Must correspond to a trained model with the same name 
    data_dir: str 
        Path to directory containing input data used in building CNN 
    model_output_dir: str
        Path to directory containing saved output from trained CNN. 
        Requires that saved trained model exists in directory [model_output_dir] with the name "[model_id]_model/"
        Requires that saved model predictions exists in directory [model_output_dir] with the name "[model_id]_predictions.csv"
    figures_dir: str 
        Path to directory for saving figures 
    num_clusters: int 
        Number of clusters to use for k-means clustering algorithm 

    Returns 
    -------
    None 
    
    """

    # Make figures directory if it doesn't exist 
    check_and_create_dir(figures_dir)
    
    # Get model settings for MODEL_ID 
    settings = get_model_settings(model_id)

    # ------------ READ IN MODEL INPUT DATA AND TRAINED MODEL ------------

    print("Reading in model input data...")
    # Get formatted input data 
    x_train_ds, y_train_df, x_val_ds, y_val_df, x_test_ds, y_test_df = get_input_data(DATA_DIR, settings)
    x_train, y_train, y_train_onehot, x_val, y_val, y_val_onehot, x_test, y_test, y_test_onehot = format_input_data(x_train_ds, y_train_df, x_val_ds, y_val_df, x_test_ds, y_test_df)

    # Combine training, validation, and testing datasets into one object 
    x_ds = xr.concat(
        [x_train_ds, x_val_ds, x_test_ds], 
        dim="time"
        )
    x = x_ds.to_array().transpose("time", "lat", "lon", "variable").values

    # Get names of features 
    feature_ids = list(x_ds.data_vars)

    # Dictionary with description of each feature for annotating plots
    features_dict = { 
        "hgt_detrended_anom": "Geopotential Height Anomalies",
        "slp_anom": "Sea Level Pressure Anomalies"
    }

    print("Reading in trained model...")
    # Read in trained model
    model_path = "{0}{1}_model/".format(MODEL_OUTPUT_DIR,MODEL_ID)
    model =  tf.keras.models.load_model(model_path)

    # ------------ READ IN MODEL PREDICTIONS ------------

    print("Reading in saved model predictions...")
    predictions_filepath = "{0}{1}_predictions.csv".format(MODEL_OUTPUT_DIR,MODEL_ID)
    predictions = pd.read_csv(predictions_filepath, index_col=False)
    dates_epcp = predictions[predictions["predicted_class"]==1]["time"].values # Days when EPCP is predicted to occur
    dates_no_epcp = predictions[predictions["predicted_class"]==0]["time"].values # Days when EPCP is NOT predicted to occur

    # ------------ COMPUTE THE LAYERWISE RELEVANCE PROPAGATION INTERPRETATIONS ------------
    # Use the package innvestigate (https://github.com/albermax/innvestigate) to build the LRP maps 

    print("Computing LRP interpretations...")
    # Remove the last dense layer with softmax activation
    model_no_softmax = innvestigate.model_wo_softmax(model)

    # Build the LRP analyzer 
    lrp_analyzer = innvestigate.analyzer.relevance_based.relevance_analyzer.LRPAlpha1Beta0(model_no_softmax)

    # Use the analyzer 
    lrp_np = lrp_analyzer.analyze(x)

    # Put numpy array into an xarray object 
    lrp_ds = xr.Dataset(
        data_vars=dict(
            {
                feature_ids[0]:(["time", "lat", "lon"], lrp_np[:,:,:,0]), 
                feature_ids[1]:(["time", "lat", "lon"], lrp_np[:,:,:,1]),
            }
        ),
        coords=x_ds.coords
    )

    # Get mean LRP on dates in which the model predicts no EPCP (y=0)
    lrp_no_epcp = lrp_ds.sel(time=dates_no_epcp)
    mean_lrp_no_epcp = lrp_no_epcp.mean(dim="time").to_array()

    # Get mean LRP on dates in which the model predicts EPCP (y=1)
    lrp_epcp = lrp_ds.sel(time=dates_epcp)
    mean_lrp_epcp = lrp_epcp.mean(dim="time").to_array()

    # ------------ MAKE LRP FIGURES ------------ 
    print("Making and saving LRP figures...")

    cbar_label = "normalized relevance (unitless)"
    cmap = "viridis"
    levels = 15
    
    # LRP heatmaps for EPCP days 
    make_heatmap_by_col(
        mean_lrp_epcp, 
        col="variable", 
        cmap=cmap,
        levels=levels,
        cbar_label=cbar_label,
        title=STATE+" EPCP days: Time-averaged LRP maps", 
        figures_dir=figures_dir, 
        savefig=True, 
        figname="lrp_epcp"
    )

    # LRP heatmaps for non-EPCP days 
    make_heatmap_by_col(
        mean_lrp_no_epcp, 
        col="variable", 
        cbar_label=cbar_label,
        cmap=cmap,
        levels=levels,
        title=STATE+" Non-EPCP days: Time-averaged LRP maps", 
        figures_dir=figures_dir, 
        savefig=True, 
        figname="lrp_non-epcp"
        )

    # ------------ COMPUTE KMEANS CLUSTERING ON THE LRP INTERPRETATIONS ------------ 
    # Apply kmeans_clusters_xr function across each variable in the dataset 
    # i.e. computer cluster center for each input feature 
    print("Peforming k-means clustering with {0} clusters...".format(num_clusters))
    
    # Cluster centers for EPCP days 
    clusters_epcp_ds = kmeans_clusters_xr(lrp_epcp.to_array(), num_clusters=num_clusters)
    
    # Cluster centers for non-EPCP days 
    clusters_no_epcp_ds = kmeans_clusters_xr(lrp_no_epcp.to_array(), num_clusters=num_clusters)

    # ------------ MAKE CLUSTERS FIGURES ------------ 
    print("Making and saving kmeans cluster figures...")

    cbar_label = "normalized relevance (unitless)"
    levels = 30
    
    # Clusters for EPCP days 
    pl = make_heatmap_by_col(
        clusters_epcp_ds, 
        col="cluster", 
        cbar_label=cbar_label,
        levels=levels, 
        title=STATE+" EPCP days", 
        figures_dir=figures_dir, 
        savefig=True, 
        figname="clusters_epcp"
        )
    # Clusters for non-EPCP days 
    pl = make_heatmap_by_col(
        clusters_no_epcp_ds, 
        col="cluster", 
        cbar_label=cbar_label, 
        levels=levels,
        title=STATE+" Non-EPCP days", 
        figures_dir=figures_dir, 
        savefig=True, 
        figname="clusters_non-epcp"
        )
        
    print("Complete!")

if __name__ == "__main__":
    main()