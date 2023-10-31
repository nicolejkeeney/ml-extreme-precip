# Helper functions related to Random Forest

import pandas as pd
import numpy as np
from sklearn import metrics


def rf_metrics(y_target, y_predicted, print_console=True):
    """Output useful metrics for interpretting results
    Returns results as a list of floats
    Prints rounded results to console

    Returns:
        accuracy (float)
        recall (float)
        precision (float)

    """
    accuracy = metrics.accuracy_score(y_target, y_predicted)
    recall = metrics.recall_score(y_target, y_predicted)
    precision = metrics.precision_score(y_target, y_predicted)

    # Print to console
    if print_console:
        print("Accuracy: ", np.around(accuracy * 100), "%")
        print("Recall: ", np.around(recall * 100), "%")
        print("Precision: ", np.around(precision * 100), "%")

    return (accuracy, recall, precision)


def confusion_matrix(y_target, y_predicted):
    """Build confusion matrix for input data"""

    def rate(data):
        """Compute rate at which value is detected"""
        if len(data) == 0:
            return 0
        else:
            return 100 * np.sum(data) / (data).size

    # When the model predicts extreme precip, but that's false
    false_positive = y_target[y_predicted == 1] == 0

    # When the model predicts NO extreme precip, but there actually WAS extreme precip
    false_negative = y_target[y_predicted == 0] == 1

    # When the model correctly predicts extreme precip
    true_positive = y_target[y_predicted == 1] == 1

    # When the model correctly predicts NO extreme precip
    true_negative = y_target[y_predicted == 0] == 0

    conf_matrix = [
        [rate(true_negative), rate(false_positive)],
        [rate(false_negative), rate(true_positive)],
    ]
    conf_matrix_key = [
        ["True Negative", "False Positive"],
        ["False Negative", "True Positive"],
    ]
    conf_matrix_df = pd.DataFrame(conf_matrix, columns=["0", "1"], index=["0", "1"])
    return conf_matrix_df


def style_cm(df):
    """Style DataFrame as confusion matrix

    References
    ----------
    https://pandas.pydata.org/docs/user_guide/style.html

    """
    styler = df.style
    styler.set_caption("Confusion Matrix")
    styler.set_table_styles(
        [  # create internal CSS classes
            {"selector": ".true", "props": "background-color: #e6ffe6;"},
            {"selector": ".false", "props": "background-color: #ffe6e6;"},
        ],
        overwrite=False,
    )
    cell_color = pd.DataFrame(
        [["true ", "false "], ["false ", "true "]], index=df.index, columns=df.columns
    )
    styler.set_td_classes(cell_color)
    return styler


def confusion_matrix_key():
    """Make confusion matrix key"""
    df = pd.DataFrame(
        [["True Negative", "False Positive"], ["False Negative", "True Positive"]],
        columns=["Negative", "Positive"],
        index=["Negative", "Positive"],
    )
    return df
