"""Helper functions for building, training, and tuning model 

Author: 
    Nicole Keeney 

Version: 
    11-06-2023


"""
import numpy as np 
import pandas as pd
import plotly.figure_factory as ff
from sklearn.utils.class_weight import compute_class_weight 

# Defaults 
OUTPUT_DIR = "" # Working directory 
FIGURES_DIR = "" # Working directory
MODEL_ID = ""  

def get_class_weights(y): 
    """ Compute class weights. Uses sklearn function compute_class_weight 
    Class 0: no extreme precip
    Class 1: extreme precip
    Class weights are roughly inversely proportional to frequency of each class.

    Parameters
    ----------
    y: np.array 
        Data to compute class weights for 
        Must be 1D array (not one hot encoded)

    Returns 
    -------
    class_weights: dict
        Dictionary of class weight by class {0: weight, 1: weight}

    References 
    ----------
    https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

    """
    classes = [0,1]
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array(classes), y=y
    )
    class_weights = {classes[0]: class_weights[0], classes[1]: class_weights[1]}
    return class_weights 


def compute_model_metrics(model, x_train, y_train_onehot, x_val, y_val_onehot, x_test, y_test_onehot, save_to_png=True, save_to_csv=True, output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR, model_id=MODEL_ID): 
    """Compute model metrics for training, validation, and testing data 
    Returns a pandas dataframe with columns for model metrics: loss, accuracy, precision, recall, and AUC 
    
    Parameters
    -----------
    model: tensorflow model 
    x_train: np.array 
        Features for training data (4D array)
        Shape of array is (time, lat, lon, variable)
    y_train_onehot: np.array 
        Labels for training data, one hot encoded (2D array)
    x_val: np.array
        Features for validation data 
    y_val_onehot: np.array
        One hot encoded labels for validation data 
    x_test: np.array
        Features for testing data 
    y_test_onehot: np.array
        One hot encoded labels for testing data 
    model_metrics: pd.DataFrame 
        Model metrics 
    save_to_png: boolean, optional 
        Save table as png file? Default to TRUE 
    save_to_csv: boolean, optional 
        Save table as csv file? Default to TRUE 
    output_dir: str, optional 
        Directory for saving model metrics csv file
        Default to working directory 
    figures_dir: str, optional 
        Directory for saving model metrics png file
        Default to working directory. 
    model_id: str, optional 
        String identifier for model

    Returns
    -------
    model_metrics: pd.DataFrame 
        Model metrics 

    References 
    -----------
    https://www.tensorflow.org/guide/keras/training_with_built_in_methods
    """
    verbose = 0
    loss_train, acc_train, prec_train, recall_train, auc_train = model.evaluate(
        x_train,
        y_train_onehot,
        verbose=verbose,
    )
    loss_val, acc_val, prec_val, recall_val, auc_val = model.evaluate(
        x_val, 
        y_val_onehot, 
        verbose=verbose
    )
    loss_test, acc_test, prec_test, recall_test, auc_test = model.evaluate(
        x_test,
        y_test_onehot,
        verbose=verbose
    )
    model_metrics = pd.DataFrame(
        {
            "Loss": [loss_train, loss_val, loss_test],
            "Accuracy": [acc_train, acc_val, acc_test],
            "Precision": [prec_train, prec_val, prec_test],
            "Recall": [recall_train, recall_val, recall_test],
            "AUC": [auc_train, auc_val, auc_test],
            "dataset": ["training", "validation", "testing"]
        }
    )
    if save_to_png or save_to_csv: 
        _save_model_metrics(
            model_metrics, 
            save_to_png=save_to_png, 
            save_to_csv=save_to_csv, 
            output_dir=output_dir, 
            figures_dir=figures_dir, 
            model_id=model_id
        )
    return model_metrics 

def _save_model_metrics(model_metrics, save_to_png=True, save_to_csv=True, output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR, model_id=MODEL_ID):
    """Save model metrics to image and csv file 
    Files will be saved to directory RESULTS_DIR 

    Parameters 
    ----------
    model_metrics: pd.DataFrame 
        Model metrics 
    save_to_png: boolean, optional 
        Save table as png file? Default to TRUE 
    save_to_csv: boolean, optional 
        Save table as csv file? Default to TRUE 
    output_dir: str, optional 
        Directory for saving model metrics csv file
    figures_dir: str, optional 
        Directory for saving model metrics png file
    model_id: str, optional 
        String identifier for model

    Returns 
    -------
    None 

    """ 

    fig_name = "{0}_model_metrics".format(model_id) if model_id != "" else "model_metrics"

    # Save pretty table as png
    if save_to_png: 
        fig = ff.create_table(model_metrics.round(3))
        fig.update_layout(
            autosize=False,
            width=500,
            height=200,
        )
        fig.write_image(figures_dir+fig_name+".png", scale=3)

    # Save as csv
    if save_to_csv: 
        model_metrics.round(3).to_csv(output_dir+fig_name+".csv", index=False)

    return None 


def _get_class_predictions_df(predictions): 
    """Assign class predicions.

    Parameters 
    ----------
    predictions: 2D np.array 
        Probability predictions from model 

    Returns
    -------
    P_all_df: pd.DataFrame 
        Predictions for all datasets
    
    """
    predict_df = pd.DataFrame(predictions)
    predict_df = predict_df.rename(columns = {0: 'prob_0', 1: 'prob_1'})
    predict_df['predicted_class'] = np.argmax(predictions, axis=1)
    return predict_df

def make_model_predictions(model, x_train, x_val, x_test, y_train_df=None,  y_val_df=None, y_test_df=None, save_to_csv=True, output_dir=OUTPUT_DIR, model_id=MODEL_ID): 
    """Make predictions for training, validation, and testing datasets using trained model. 
    If y_train_df, y_val_df, and y_test_df are provided as arguments, they will be added to the final DataFrame. 
    y_train_df, y_val_df, and y_test_df MUST have a time column 
    Time column will only be added to final DataFrame if y_train_df, y_val_df, and y_test_df are provided. 

    Parameters
    ----------
    x_train: np.array 
        Features for training data (4D array)
        Shape of array is (time, lat, lon, variable)
    x_val: np.array
        Features for validation data 
    x_test: np.array
        Features for testing data 
    y_train_df: pd.DataFrame 
        Labels for training data  + any additional columns of interest (i.e. precip mean) to add to table
    y_val_df: pd.DataFrame, optional
        Labels for validation data  + any additional columns of interest (i.e. precip mean) to add to table
    y_test_df: pd.DataFrame, optional 
        Labels for testing data + any additional columns of interest (i.e. precip mean) to add to table
    save_to_csv: boolean, optional 
        Save table as csv file? Default to TRUE 

    Returns 
    -------
    P_all_df: pd.DataFrame 
        Predictions and labels by timestep.
        Table columns: prob_0, prob_1, predicted_class, set, + any columns in y_train_df, y_val_df, and y_test_df 
    """

    # Make predictions for each dataset 
    P_train = model.predict(x_train)
    P_val = model.predict(x_val)
    P_test = model.predict(x_test)

    # Get the predicted class from the probabilties 
    P_train_df = _get_class_predictions_df(P_train)
    P_train_df["set"] = ["training"]*len(P_train_df)

    P_val_df = _get_class_predictions_df(P_val)
    P_val_df["set"] = ["validation"]*len(P_val_df)

    P_test_df = _get_class_predictions_df(P_test)
    P_test_df["set"] = ["testing"]*len(P_test_df)

    # Concatenate all the datasets into one dataframe
    P_all_df = pd.concat([P_train_df, P_val_df, P_test_df])

    # Add in labels data and merge on time column 
    if (y_train_df is not None) and (y_val_df is not None) and (y_test_df is not None): 
        y_all_df = pd.concat([y_train_df, y_val_df, y_test_df]) 
        P_all_df["time"] = y_all_df["time"]
        P_all_df = y_all_df.merge(P_all_df, on="time")
        P_all_df["time"] = pd.to_datetime(P_all_df["time"].values) # Convert time values to datetime objects 
        #P_all_df = P_all_df.set_index("time") # Set time column to index 

    # Rename precip_classes column for better readibility in plots 
    P_all_df = P_all_df.rename(columns={"precip_classes": "true_class"})

    # Save to csv 
    if save_to_csv: 
        predictions_path = output_dir + model_id + "_predictions.csv"
        P_all_df.to_csv(predictions_path, index=False) 

    return P_all_df 