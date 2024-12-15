import torch
import torch.nn as nn
from typing import Callable, Union
from collections import OrderedDict
from abc import abstractmethod

from .utils.nn_utils import get_activation, Conv3DSame
from .base_model import BaseModel

class SFCN(BaseModel):
    """ Base Simple Fully Convolutional Network (SFCN) model. Adapted
    from https://doi.org/10.1016/j.media.2020.101871. A simple, VGG-like
    convolutional neural network for 3-dimensional neuroimaging data.
    """

    FILTERS = [32, 64, 128, 256, 256, 64]

    @abstractmethod
    def prediction_head(cls, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Base SFCN model has no prediction head and '
                                  'should not be initialized with '
                                  'include_top=True')

    def __init__(self, *,
                 in_channels: int = 1,
                 pooling: Union[str, nn.Module] = 'avg',
                 include_top: bool = False,
                 dropout: float = 0.0,
                 weight_decay: float = 0.0,
                 activation: Union[str, nn.Module] = 'relu',
                 filters: list[int] = FILTERS,
                 weights: str = None,
                 eps: float = 0.001,
                 name: str = 'sfcn',
                 **kwargs):
        """
        Args:
            input_shape (tuple): Input shape of the tensor (C, D, H, W).
            pooling (str): Pooling type ('avg' or 'max').
            include_top (bool): If True, include a prediction head.
            dropout (float): Dropout rate.
            weight_decay: Control the regularisation of Conv3D.
            activation (str): Activation function ('relu', 'tanh', etc.).
            filters (list): Custom filter sizes for each block.
        """
        super(SFCN, self).__init__()

        self.include_top = include_top
        self.name = name
        self.weight_decay = weight_decay
        self.eps = eps

        # Define layers
        channel_dict = OrderedDict()

        self._build_conv_blocks(filters, in_channels, activation, channel_dict)
        channel_dict[f'{name}_top_conv'] = Conv3DSame(
            filters[-2], filters[-1], kernel_size=1
        )
        channel_dict[f'{name}_top_norm'] = nn.BatchNorm3d(filters[-1], eps=self.eps)
        channel_dict[f'{name}_top_{activation}'] = get_activation(activation)
        channel_dict[f'{name}_top_pool'] = self.get_global_pooling_layer(pooling)

        if include_top:
            channel_dict[f'{name}_top_dropout'] = nn.Dropout(dropout)

        self.fn1 = nn.Sequential(channel_dict)
        self.out_channels: int = filters[-1]

    def _build_conv_blocks(self, filters: list[int], in_channels: int, activation: str, channel_dict: OrderedDict):
        curr_filters = in_channels
        for i in range(len(filters) - 1):
            out_channels = filters[i]
            block_name = f'{self.name}_block{i+1}'
            channel_dict[f'{block_name}_conv'] = Conv3DSame(
                    curr_filters,
                    out_channels,
                    kernel_size=3,
                    bias=True
                )
            channel_dict[f'{block_name}_norm'] = nn.BatchNorm3d(out_channels, eps=self.eps)
            channel_dict[f'{block_name}_{activation}'] = get_activation(activation)
            channel_dict[f'{block_name}_pool'] = nn.MaxPool3d(kernel_size=2, stride=2)

            curr_filters=filters[i]

    def get_global_pooling_layer(self, pooling: Union[str, Callable]) -> nn.Module:
        if callable(pooling):
            return pooling
        if pooling == "avg":
            return nn.AdaptiveAvgPool3d(1)
        elif pooling == "max":
            return nn.AdaptiveMaxPool3d(1)
        else:
            raise ValueError(f"Unsupported pooling type: {pooling}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fn1(x)
        if self.include_top: 
            x = self.prediction_head(x)
        return x