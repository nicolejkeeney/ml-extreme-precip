"""Train model

Author:
    Nicole Keeney 
    
Version: 
    10-30-2023 

"""

import pandas as pd
import numpy as np
import time
import tensorflow as tf 
from tqdm.keras import TqdmCallback
from model_architecture import get_compiled_model
from utils.misc_utils import (
    get_model_settings, 
    check_and_create_dir
)  
from utils.read_data_utils import (
    get_input_data, 
    format_input_data
)
from utils.cnn_utils import (
    get_class_weights,
    compute_model_metrics,
    make_model_predictions
)

# ------------ SET GLOBAL VARIABLES ------------
 
# Model ID needs to match string key in model_settings.json
MODEL_ID = "frances_Missouri"

# Directories. Needs to have a slash (/) after (i.e "dir/")
DATA_DIR = "../data/input_data_preprocessed/us_states/" 
OUTPUT_DIR = "../model_output/us_states/"+MODEL_ID+"/" 
FIGURES_DIR = "../figures/us_states/"+MODEL_ID+"/" 

# Settings 
SAVE_TRAINING_HISTORY = True 
SAVE_MODEL = True # save trained tensorflow model? 
SAVE_MODEL_METRICS = True # metrics will be saved as a png to figures_dir and a csv to output_dir 
SAVE_PREDICTIONS = True 

# ------------ MAIN FUNCTION ------------

def main(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR, model_id=MODEL_ID, save_training_history=SAVE_TRAINING_HISTORY, save_model=SAVE_MODEL, save_model_metrics=SAVE_MODEL_METRICS, save_predictions=SAVE_PREDICTIONS): 
    """Build and train CNN 
    Argument defaults to global variables in module 
    
    Parameters 
    ----------
    data_dir: str, optional 
        Directory for preprocessed input data 
    output_dir: str, optional 
        Directory for saving trained model, training history, model metrics
    figures_dir: str, optional 
        Directory for saving figures
    model_id: str, optional 
        String identifier for model. Must match string key in model_settings.json
    save_training_history: boolean, optional 
        Save trainig history as csv file? Default to True 
    save_model: boolean, optional 
        Save trained tensorflow model? Default to True 
    save_model_metrics: boolean, optional 
        Save model metrics? Default to true 
        Metrics will be saved as a png to figures_dir and a csv to output_dir 
    save_predictions: boolean, optional 
        Save model predictions as a csv file? Default to true

    Returns 
    -------
    model
        Trained CNN 
    predictions: pd.DataFrame
        Predictions and labels by timestep.
        Table columns: prob_0, prob_1, predicted_class, set, + any columns in y_train_df, y_val_df, and y_test_df 
    """
    start_time = time.time()
    check_and_create_dir([output_dir, figures_dir])
    print("Model output will be saved to: {0}".format(output_dir))
    print("Figures will be saved to: {0}".format(figures_dir))

    # Get model settings 
    settings = get_model_settings(model_id)

    # Get formatted input data 
    print("Retrieving and formatting input data...")
    x_train_ds, y_train_df, x_val_ds, y_val_df, x_test_ds, y_test_df = get_input_data(data_dir, settings)
    x_train, y_train, y_train_onehot, x_val, y_val, y_val_onehot, x_test, y_test, y_test_onehot = format_input_data(x_train_ds, y_train_df, x_val_ds, y_val_df, x_test_ds, y_test_df)

    # Get class weights for training data 
    class_weights = get_class_weights(y_train)
    settings["class_weights"] = class_weights  # Add to model settings
    print("Class weights: {0}".format(class_weights))

    # Build and train model 
    print("Building and training model...")
    model = compile_and_train_cnn(
        settings, 
        x_train, y_train_onehot, x_val, y_val_onehot, 
        early_stop=True, 
        save_history=save_training_history, 
        save_model=save_model, 
        output_dir=output_dir,
        model_id=model_id
        )
    print(model.summary())
    
    # Compute model metrics and save to png and csv 
    print("Computing model metrics...")
    model_metrics_df = compute_model_metrics(
        model, 
        x_train, y_train_onehot, x_val, y_val_onehot, x_test, y_test_onehot, 
        save_to_png=save_model_metrics, 
        save_to_csv=save_model_metrics, 
        output_dir=output_dir, 
        figures_dir=figures_dir, 
        model_id=model_id
    )
    print(model_metrics_df)

    # Make predictions and save to csv 
    print("Making model predictions...") 
    predictions = make_model_predictions(
        model, 
        x_train, x_val, x_test, y_train_df,  y_val_df, y_test_df, 
        save_to_csv=save_predictions, 
        output_dir=output_dir, 
        model_id=model_id
    )

    print("Building and training complete!")
    time_elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print("TOTAL TIME ELAPSED: {0}".format(time_elapsed))

    return model, predictions

def compile_and_train_cnn(settings, x_train, y_train_onehot, x_val, y_val_onehot, early_stop=True, save_history=True, save_model=True, output_dir=OUTPUT_DIR, model_id=MODEL_ID):
    """ Build and train convolutional neural network using input settings 

    Parameters 
    ---------
    settings: dict 
        Model settings 
    x_train: np.array 
        Features for training data (4D array)
        Shape of array is (time, lat, lon, variable)
    y_train_onehot: np.array 
        Labels for training data, one hot encoded (2D array)
    x_val: np.array 
        Features for validation data (4D array)
    y_val_onehot: np.array
        Labels for training data, one hot encoded (2D array)
    early_stop: boolean, optional 
        Do early stopping during training? Default to TRUE
    output_dir: str, optional 
        Directory for saving model. Default to current directory 
    model_id: str, optional 
        String identifier for model 
    save_history: boolean, optional 
        Save traning history to csv file? Default to TRUE
        File will be saved to directory RESULTS_DIR 
    save_model: boolean, optional 
        Save trained model? Default to TRUE 
        File will be saved to directory MODELS_DIR 
    
    Returns 
    -------
    model 

    """
    # Print a pretty progress bar during model training 
    callbacks = [TqdmCallback(verbose=0)] 

    # Add early stopping to settings dict
    if early_stop: 
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=10, 
            restore_best_weights=True, 
            start_from_epoch=10
        )
        callbacks.append([early_stopping])

    model = get_compiled_model(
        data_shape=x_train.shape[1:], # (lat, lon, number of classes)
        learning_rate=settings["learning_rate"],
        conv_filters=settings["conv_filters"],
        dense_layers=settings["dense_layers"],
        dense_neurons=settings["dense_neurons"],
        dropout_rate=settings["dropout_rate"],
        activity_reg=settings["activity_reg"],
        random_seed=settings["random_seed"],
    )

    # Train model 
    history = model.fit(
        x_train,
        y_train_onehot,
        batch_size=settings["batch_size"],
        epochs=settings["epochs"],
        class_weight=settings["class_weights"],
        validation_data=(x_val, y_val_onehot),
        callbacks=callbacks,
        verbose=0,
    )

    # Save model training history as csv file
    if save_history: 
        history_path = output_dir + model_id + "_history.csv"
        hist_df = pd.DataFrame(history.history)
        hist_df["epoch"] = hist_df.index
        hist_df.to_csv(history_path, index=False)

    # Save the TF model 
    if save_model: 
        model_dir = output_dir + model_id + "_model"
        tf.keras.models.save_model(model, model_dir, overwrite=True)

    return model 

if __name__ == "__main__":
    main()