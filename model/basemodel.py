
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import numpy as np
import typing as tp

import optuna

from utils.io_utils import save_predictions_to_file, save_model_to_file


class BaseModel:
    """Basic interface for all models.

    All implemented models should inherit from this base class to provide a common interface.
    At least they have to extend the init method defining the model and the define_trial_parameters method
    specifying the hyperparameters.

    Methods
    -------
    __init__(params, args):
        Defines the model architecture, depending on the hyperparameters (params) and command line arguments (args).
    fit(X, y, X_val=None, y_val=None)
        Trains the model on the trainings dataset (X, y). Validates the training process and uses early stopping
        if a validation set (X_val, y_val) is provided. Returns the loss history and validation loss history.
    predict(X)
        Predicts the labels of the test dataset (X). Saves and returns the predictions.
    attribute(X, y)
        Extract feature attributions for input pair (X, y)
    define_trial_parameters(trial, args)
        Returns a possible hyperparameter configuration. This method is necessary for the automated hyperparameter
        optimization.
    save_model_and_prediction(y_true, filename_extension="")
        Saves the current state of the model and the predictions and true labels of the test dataset.
    save_model(filename_extension="")
        Saves the current state of the model.
    save_predictions(y_true, filename_extension="")
        Saves the predictions and true labels of the test dataset.
    clone()
        Creates a fresh copy of the model using the same parameters, but ignoring any trained weights. This method
        is necessary for the cross validation.
    """

    def __init__(self, params: tp.Dict = None, args=None):