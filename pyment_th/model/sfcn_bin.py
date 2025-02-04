""" Module containing the binary SFCN implementation. """
import torch
import torch.nn as nn

from collections import OrderedDict

from .sfcn import SFCN

class BinarySFCN(SFCN):

    def prediction_head(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view((x.shape[0], -1))
        x = self.linear(x)
        return self.act(x)


    def __init__(self, 
                 *args,
                 include_top: bool = True,
                 name: str = 'sfcn-bin',
                 **kwargs):

        super().__init__(*args, include_top=include_top, name=name, **kwargs)
        
        self.linear = nn.Sequential(OrderedDict({f'{name}_predictions': nn.Linear(self.out_channels, 1)}))
        self.act = nn.Sequential(OrderedDict({f'{name}_sigmoid': nn.Sigmoid()}))
