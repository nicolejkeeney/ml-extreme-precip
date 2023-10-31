# Helper functions for convolutional neural network model

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import regularizers


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


def _get_uncompiled_model(
    data_shape=(15, 35, 2),
    conv_filters=16,
    dense_neurons=16,
    dense_layers=1,
    activity_reg=0.001,
    dropout_rate=0.2,
    random_seed=25,
):
    """Build uncompiled CNN

    Parameters
    ----------
    data_shape: tuple
        Shape of data in the form (lat, lon, num_classes)
    conv_filters: int
        Number of filters to use in convolution layer
    dense_neurons: int
        Number of dense neurons to use in dense layer
    dense_layers: int
        Number of dense layers to use
    activity_reg: float
        Regularization factor for l2 regularization
    dropout_rate: float
        Dropout rate
    random_seed: int
        Random seed for initializing bias & kernel in each layer

    Returns
    -------
    keras.engine.sequential.Sequential
        Uncompiled model

    """
    model = models.Sequential()
    model.add(layers.Input(shape=data_shape))  ## define input shape

    # Convolutional and pooling layers are the feature extraction portion of the network
    model.add(
        layers.Conv2D(
            conv_filters,
            (3, 3),
            activity_regularizer=regularizers.l2(activity_reg),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
        )
    )
    model.add(layers.Activation("relu"))  ## add convolutional layer
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(dropout_rate))

    model.add(
        layers.Conv2D(
            conv_filters,
            (3, 3),
            activity_regularizer=regularizers.l2(activity_reg),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed + 1),
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed + 1),
        )
    )
    model.add(layers.Activation("relu"))  ## add convolutional layer
    model.add(layers.MaxPooling2D((2, 2)))  ## pooling layer
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Flatten())  ## converts from 2D array to 1D array

    # Feature interpretation layers
    for i in range(dense_layers):  ## add dense layer
        model.add(
            layers.Dense(
                dense_neurons,
                activity_regularizer=regularizers.l2(activity_reg),
                bias_initializer=tf.keras.initializers.RandomNormal(
                    seed=random_seed + 2
                ),
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    seed=random_seed + 2
                ),
            )
        )
        model.add(layers.Activation("relu"))

    model.add(
        layers.Dense(2, activation="softmax")
    )  ## classifier layer (binary class where 1=extreme)
    return model


def get_compiled_model(data_shape=(15, 35, 2), learning_rate=0.0004, **kwargs):
    """Get incompiled model, and then.... compile it

    Parameters
    ----------
    lr: float
        Learning rate

    Returns
    --------
    keras.engine.sequential.Sequential
        Compiled model
    """
    # Get uncompiled model
    model = _get_uncompiled_model(data_shape=data_shape, **kwargs)

    # Set metrics
    # https://stackoverflow.com/questions/58630393/does-tf-keras-metrics-auc-work-on-multi-class-problems
    # https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC
    METRICS = [
        metrics.CategoricalAccuracy(name="accuracy"),
        metrics.Precision(class_id=1, name="precision"),
        metrics.Recall(class_id=1, name="recall"),
        tf.keras.metrics.AUC(curve="ROC", multi_label=False),
    ]

    # Compile model
    model.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=METRICS,
    )
    return model


def build_kt_tuning_model(hp):
    """Build and compile CNN with tuning params

    Parameters
    ----------
    None

    Returns
    --------
    keras.engine.sequential.Sequential
        Compiled model
    """
    # Define hyperparameters to optimize
    learning_rate = hp.Float(
        "learning_rate", min_value=0.0001, max_value=0.01, step=10, sampling="log"
    )
    activity_reg = hp.Float(
        "activity_reg_factor", min_value=0.0001, max_value=0.1, step=10, sampling="log"
    )
    conv_filters = hp.Int("num_conv_filters", min_value=8, max_value=32, step=8)
    dense_neurons = hp.Int("num_dense_neurons", min_value=8, max_value=32, step=8)
    dense_layers = hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)
    random_seed = hp.Int("random_seed", min_value=25, max_value=200, step=5)
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)

    # Build model
    model = models.Sequential()
    model.add(layers.Input(shape=(15, 35, 2)))  ## define input shape

    # Convolutional layers
    model.add(
        layers.Conv2D(
            conv_filters,
            (3, 3),
            activity_regularizer=regularizers.l2(activity_reg),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed),
        )
    )
    model.add(layers.Activation("relu"))  ## add convolutional layer
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(
        layers.Conv2D(
            conv_filters,
            (3, 3),
            activity_regularizer=regularizers.l2(activity_reg),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=random_seed + 1),
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=random_seed + 1),
        )
    )
    model.add(layers.Activation("relu"))  ## add convolutional layer
    model.add(layers.MaxPooling2D((2, 2)))  ## pooling layer
    model.add(layers.Dropout(dropout_rate))

    # Dense layers
    model.add(layers.Flatten())  ## converts from 2D array to 1D array

    # Feature interpretation layers
    for i in range(dense_layers):
        model.add(
            layers.Dense(
                dense_neurons,
                activity_regularizer=regularizers.l2(activity_reg),
                bias_initializer=tf.keras.initializers.RandomNormal(
                    seed=random_seed + 2
                ),
                kernel_initializer=tf.keras.initializers.RandomNormal(
                    seed=random_seed + 2
                ),
            )
        )  ## dense layer
        model.add(layers.Activation("relu"))
    model.add(
        layers.Dense(2, activation="softmax")
    )  ## classifier layer (binary class where 1=extreme)

    # Compile model
    METRICS = [
        metrics.CategoricalAccuracy(name="accuracy"),
        metrics.Precision(class_id=1, name="precision"),
        metrics.Recall(class_id=1, name="recall"),
        tf.keras.metrics.AUC(curve="ROC", multi_label=False),
    ]
    model.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        metrics=METRICS,
    )
    return model
