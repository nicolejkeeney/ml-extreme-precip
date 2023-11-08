""" Use keras tuner to do hyperparameter tuning 

Author:
    Nicole Keeney 
    
Version: 
    11-06-2023

References: 
    https://www.tensorflow.org/tutorials/keras/keras_tuner

"""

import time
import xarray as xr 
import pandas as pd
import json 
import numpy as np 
from tqdm import tqdm
import keras_tuner as kt
import tensorflow as tf 
import sys 

# Import helper functions 
from model_architecture import build_kt_tuning_model
from train_model import compile_and_train_cnn
from utils.read_data_utils import (
    get_input_data, 
    format_input_data
)
from utils.misc_utils import (
    check_and_create_dir, 
    add_to_settings_json
)
from utils.cnn_utils import (
    get_class_weights,
    compute_model_metrics, 
    make_model_predictions
)
from make_figures import (
    make_precip_vs_prob_plot,
    plot_metrics_by_epoch,
    make_epcp_composite_plot
)

# ------------ SET GLOBAL VARIABLES ------------

# Model settings 
LABELS_GEOM = "Colorado"
FEATURES_GEOM = "CONUS"
EPOCHS = 500
BATCH_SIZE = 2048

# Assign a model ID based on settings 
# Will be elongated to include trial # 
MODEL_ID = "keras_tuner_hp_"+LABELS_GEOM

# Directories. Needs to have a slash (/)
DATA_DIR = "../data/input_data_preprocessed/" 
OUTPUT_DIR = "../model_output/"
FIGURES_DIR = "../figures/"
TUNER_DIR = "../keras_tuner/"

# Json file for storing best hyperparameters 
SETTINGS_PATH = "model_settings.json"

# Save options 
SAVE_MODEL = False # save trained tensorflow model? 
MAKE_FIGURES = True 

# Set number of trials to run 
NUM_TRIALS = 5

# Number of models to train 
# Must be larger than NUM_TRIALS 
# i.e set num_models=3 to train 3 models, with the top 1st, second, and third best hyperparameter combos 
NUM_MODELS = 5

# Build settings dictionary 
SETTINGS = {
    "labels_geom": LABELS_GEOM, 
    "features_geom": FEATURES_GEOM, 
    "batch_size": BATCH_SIZE, 
    "epochs": EPOCHS
}

# Save output to logfile instead of printing to console? 
check_and_create_dir(TUNER_DIR)
logfile = TUNER_DIR + MODEL_ID + ".log"
log = open(logfile, "w+")
sys.stdout = log

def main(settings=SETTINGS, model_id=MODEL_ID, tuner_dir=TUNER_DIR, data_dir=DATA_DIR, figures_dir=FIGURES_DIR, output_dir=OUTPUT_DIR, num_trials=NUM_TRIALS, num_models=NUM_MODELS, save_model=SAVE_MODEL, make_figures=MAKE_FIGURES, settings_path=SETTINGS_PATH): 
    """Run keras tuner to get optimal hyperparameters. 
    Train model with hyperparameters. 
    Make figures. 

    Parameters 
    ----------
    settings: dict, optional 
        Model settings
    model_id: str, optional 
        String identifier for model 
    tuner_dir: str, optional 
        Directory for saving keras tuner output 
    data_dir: str, optional 
        Directory for preprocessed input data 
    output_dir: str, optional 
        Directory for saving trained model, training history, model metrics
    figures_dir: str, optional 
        Directory for saving figures
    num_trials: int, optional 
        Number of trials to run tuner for 
    num_models: int, optional 
        Number of models to train 
    save_model: boolean, optional 
        Save trained tensorflow model? Default to True 
    make_figures: boolean, optional 
        Make model figures? Default to True 
    settings_path: str, optional 
        Path to settings json file for storing best hyperparameters

    Returns 
    -------
    None 

    """

    ## --------- SETUP ------------
    start_time = time.time()

    print("USER INPUTS\n------------")
    print("Model ID: {0}".format(model_id))
    print("Data directory: {0}".format(data_dir))
    print("Figure directory: {0}".format(figures_dir))
    print("Data output directory: {0}".format(output_dir))
    print("Keras tuner output directory: {0}".format(tuner_dir))
    print("Number of trials: {0}".format(num_trials))
    print("Number of epochs: {0}".format(settings["epochs"]))
    print("Number of models to train: {0}".format(num_models))
    print("Make figures? {0}".format(make_figures))
    print("Save trained model? {0}".format(save_model))
    print("Settings: {0}".format(settings))

    # Get formatted input data 
    print("Retrieving and formatting input data...", end="")
    x_train_ds, y_train_df, x_val_ds, y_val_df, x_test_ds, y_test_df = get_input_data(data_dir, settings)
    x_train, y_train, y_train_onehot, x_val, y_val, y_val_onehot, x_test, y_test, y_test_onehot = format_input_data(x_train_ds, y_train_df, x_val_ds, y_val_df, x_test_ds, y_test_df)
    print("complete!")

    # Get class weights for training data 
    class_weights = get_class_weights(y_train)
    settings["class_weights"] = class_weights  # Add to model settings
    #print("Class weights: {0}".format(class_weights))

    # Add early stopping to settings dict
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=10, 
        restore_best_weights=True, 
        start_from_epoch=10
    )
    callbacks = [early_stopping]

    ## --------- BUILD TUNER ------------
    # kt.GridSearch will go through all options 
    # kt.BayesianOptimization performs tuning with Gaussian process
    tuner = kt.BayesianOptimization(
        build_kt_tuning_model,
        objective=[
            kt.Objective("val_loss", direction="min"),
            kt.Objective("val_recall", direction="max")
        ],
        max_trials=num_trials,
        overwrite=True, 
        directory=tuner_dir, 
        project_name=model_id
    )

    # tuner.search_space_summary()

    ## --------- FIND BEST HYPERPARAMETERS ------------
    print("Finding the best hyperparameters...")    
    tuner.search(
        x_train, y_train_onehot,
        validation_data=(x_val, y_val_onehot), 
        epochs=settings["epochs"],  
        class_weight=class_weights, 
        callbacks=callbacks,
        verbose=1
    )
    print("Tuner search complete.")

    # Loop through the top hyperparameter sets and retrain the model 
    print("Training {0} models with the top hyperparameters".format(num_models))
    for i in tqdm(range(num_models)): 

        model_id_i = model_id+"_"+str(i)
        output_dir_i = output_dir+model_id_i+"/"
        figures_dir_i = figures_dir+model_id_i+"/"
        check_and_create_dir([output_dir_i, figures_dir_i])

        print("#{0} best hyperparameters".format(i))

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=num_models)[i]
        print(json.dumps(best_hps.values, indent=4))

        # Add to settings dict 
        settings_i = settings.copy()
        settings_i.update(best_hps.values)
        add_to_settings_json(
            m_id=model_id_i, 
            settings=settings_i, 
            settings_path=settings_path 
        )

        # Build and train model 
        print("Building and training model {0}...".format(i))
        model_i = compile_and_train_cnn(
            settings_i, 
            x_train, 
            y_train_onehot, 
            x_val, 
            y_val_onehot, 
            output_dir=output_dir_i, 
            model_id=model_id_i, 
            early_stop=True, 
            save_history=True, 
            save_model=save_model
            )
        
        # Compute model metrics and save to png and csv 
        print("Computing model metrics...")
        model_metrics_df = compute_model_metrics(
            model_i, 
            x_train, y_train_onehot, x_val, y_val_onehot, x_test, y_test_onehot, 
            save_to_png=True, 
            save_to_csv=True, 
            output_dir=output_dir_i, 
            figures_dir=figures_dir_i, 
            model_id=model_id_i
        )
        print(model_metrics_df)

        # Make predictions and save to csv 
        predictions = make_model_predictions(
            model_i, 
            x_train, x_val, x_test, y_train_df,  y_val_df, y_test_df, 
            save_to_csv=True, 
            output_dir=output_dir_i, 
            model_id=model_id_i
        )

        # Make figures 
        if make_figures: 
            ## (1) Mean precip (x) vs. predicted probability of EPCP (y)  
            make_precip_vs_prob_plot(
                predictions["precip_mean"], 
                predictions["prob_1"], 
                figures_dir=figures_dir_i, 
                figname=model_id_i+"_mean_precip_vs_pred_epcp"
            )

            ## (2) EPCP composite anomaly patterns 
            dates_epcp = predictions[predictions["predicted_class"]==1]["time"].values # Days when EPCP is predicted to occur
            x_all_ds = xr.concat([x_train_ds, x_val_ds, x_test_ds], dim="time") # Combine all the datasets into one xr.Dataset 
            x_all_epcp = x_all_ds.sel(time=dates_epcp) # Get just EPCP days
            x_epcp_mean = x_all_epcp.mean(dim="time").to_array() # Compute mean. Convert to xr.DataArray

            make_epcp_composite_plot(
                x_epcp_mean, 
                figname=model_id_i+"_epcp_composite", 
                figures_dir=figures_dir_i
            )
 
            ## (3) Metrics by epoch from model training history 
            training_history = pd.read_csv(output_dir_i + model_id_i + "_history.csv", index_col=False)
            plot_metrics_by_epoch(
                training_history, 
                figname=model_id_i+"_training_metrics_by_epoch", 
                figures_dir=figures_dir_i) 

    print("TUNER WITH {0} TRIALS AND {1} TRAINED MODELS COMPLETE".format(num_trials, num_models))
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print("TOTAL TIME ELAPSED: {0}".format(time_elapsed))
    return None


if __name__ == "__main__":
    main()