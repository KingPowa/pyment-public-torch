""" Module containing the wrapper extending the Keras Model-class. """
import os
import numpy as np
import torch.nn as nn

from .utils.etc import WeightRepository

class BaseModel(nn.Module):
    """ A model wrapper on top of the default keras Model class.
    Contains two main additions to the standard funcionality: first,
    allows the models implemented here to load weights via the
    ModelRepository. Second, bundles the appropriate postprocessing
    function with each model. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def postprocess(self, values: np.ndarray) -> np.ndarray:
    #     """ Applied the appropriate postprocessing to an array of
    #     predictions. Which postprocessing function should be applied
    #     is determined with a lookup based on the model class and the
    #     weights that are used.

    #     Parameters:
    #     -----------
    #     values : np.ndarray
    #         Raw predictions.

    #     Returns:
    #     --------
    #     np.ndarray
    #         Processed predictions.
    #     """

    #     f = get_postprocessing(modelname=self.__class__.__name__,
    #                            weights=self.weight_name)

    #     return f(values)
