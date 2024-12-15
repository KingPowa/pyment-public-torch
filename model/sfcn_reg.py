import torch
import torch.nn as nn

import numpy as np
from typing import Tuple
from collections import OrderedDict

from .sfcn import SFCN

class RegressionSFCN(SFCN):

    def prediction_head(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view((1, -1))
        x = self.linear(x)
        if self.prediction_range:
            x = x.clamp(0, self.upper-self.lower)
            x += self.lower
        return x


    def __init__(self, *args,
                 prediction_range: Tuple[int] = (3, 95),
                 include_top: bool = True,
                 name: str = 'sfcn-reg',
                 **kwargs):
        self.prediction_range = prediction_range

        super().__init__(*args, prediction_range=prediction_range,
                         include_top=include_top, name=name, **kwargs)
        
        if prediction_range is not None:
            self.lower = np.amin(prediction_range)
            self.upper = np.amax(prediction_range)
        
        self.linear = nn.Sequential(OrderedDict({f'{name}_predictions': nn.Linear(self.out_channels, 1)}))
        self.act = nn.Sequential(OrderedDict({f'{name}_restrict_relu': nn.ReLU()})) if prediction_range else None

