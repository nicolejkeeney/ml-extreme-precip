""" Module for making figures 

Author:
    Nicole Keeney 
    
Version: 
    11-03-2023

"""

import os 
import pandas as pd
import xarray as xr
from utils.misc_utils import (
    check_and_create_dir, 
    get_model_settings, 
)
from utils.read_data_utils import get_input_data 
from utils.figs_utils import (
    make_precip_vs_prob_plot,
    make_precip_vs_prob_plot_by_season,
    make_epcp_composite_plot, 
    plot_metrics_by_epoch, 

)

# ------------ SET GLOBAL VARIABLES ------------
 
# Model ID needs to match string key in model_settings.json
MODEL_ID = "frances_Florida"

# Directories. Needs to have a slash (/) after (i.e "dir/"")
DATA_DIR = "../data/input_data_preprocessed/us_states/" 
MODEL_OUTPUT_DIR = "../model_output/us_states/"+MODEL_ID+"/"  # Saved predictions should be here 
FIGURES_DIR = "../figures/us_states/"+MODEL_ID+"/" # Where to save figures to 

def main(model_id=MODEL_ID, data_dir=DATA_DIR, model_output_dir=MODEL_OUTPUT_DIR, figures_dir=FIGURES_DIR):
    """Run all figure-making functions
    
    Parameters 
    ----------
    model_id: str, optional 
        String identifier for model
    data_dir: str, optional 
        Directory for preprocessed input data 
    model_output_dir: str, optional 
        Directory where model output (predictions) has been saved
    figures_dir: str, optional 
        Directory for saving figures

    Returns 
    -------
    None 

    """

    ## -------- SETUP --------

    print("USER INPUTS\n------------")
    print("Model ID: {0}".format(model_id))
    print("Data directory: {0}".format(data_dir))
    print("Figure directory: {0}".format(figures_dir))
    print("Model output directory: {0}".format(model_output_dir))

    # Get model settings 
    settings = get_model_settings(model_id, print_to_console=False)

    # Create figures dir if it does't exist 
    check_and_create_dir(figures_dir)

    # Confirm that predictions and training history data exists 
    predictions_filepath = model_output_dir+model_id+"_predictions.csv"
    history_filepath = model_output_dir+model_id+"_history.csv"
    for pathname, path in zip(["predictions","training history"],[predictions_filepath, history_filepath]):
        if not os.path.isfile(path): 
            raise ValueError("No file found for {0} data at path {1}. Figures cannot be generated.".format(pathname, path))

    ## -------- READ DATA  --------
    
    # Saved predictions data 
    predictions = pd.read_csv(predictions_filepath, index_col=False)

    # Preprocessed model input data 
    x_train_ds, y_train_df, x_val_ds, y_val_df, x_test_ds, y_test_df = get_input_data(data_dir, settings)

    # Training history by epoch 
    training_history = pd.read_csv(model_output_dir + model_id + "_history.csv", index_col=False)
    
    ## -------- MAKE PLOTS --------
    
    ## (1) Mean precip (x) vs. predicted probability of EPCP (y)  
    make_precip_vs_prob_plot(
        predictions["precip_mean"], 
        predictions["prob_1"], 
        savefig=True, 
        figures_dir=figures_dir, 
        figname=model_id+"_mean_precip_vs_pred_epcp"
    )
    ## (2) BY SEASON: mean precip (x) vs. predicted probability of EPCP (y)  
    make_precip_vs_prob_plot_by_season(
        predictions, 
        savefig=True, 
        figures_dir=figures_dir, 
        figname=model_id+"_mean_precip_vs_pred_epcp_by_season"
        )

    ## (3) EPCP composite anomaly patterns 
    dates_epcp = predictions[predictions["predicted_class"]==1]["time"].values # Days when EPCP is predicted to occur
    x_all_ds = xr.concat([x_train_ds, x_val_ds, x_test_ds], dim="time") # Combine all the datasets into one xr.Dataset 
    x_all_epcp = x_all_ds.sel(time=dates_epcp) # Get just EPCP days
    x_epcp_mean = x_all_epcp.mean(dim="time").to_array() # Compute mean. Convert to xr.DataArray

    make_epcp_composite_plot(
        x_epcp_mean, 
        savefig=True,
        figname=model_id+"_epcp_composite", 
        figures_dir=figures_dir
    )
 
    ## (4) Metrics by epoch from model training history 
    plot_metrics_by_epoch(
        training_history, 
        savefig=True,
        figname=model_id+"_training_metrics_by_epoch", 
        figures_dir=figures_dir) 

    return None 


if __name__ == "__main__":
    main()