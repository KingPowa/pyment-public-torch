import torch.nn as nn
import torch as th
import torch.nn.functional as F

class MaxPool3dIndices(nn.MaxPool3d):

    def __init__(self, *args, **kwargs):
        super(MaxPool3dIndices, self).__init__(return_indices=True, *args, **kwargs)
        self.max_indices: dict[str, th.Tensor] = {}

    def forward(self, input: th.Tensor) -> th.Tensor:
        x, indices = super().forward(input)
        device = indices.device
        self.max_indices[device] = indices
        return x

class Conv3DSame(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, *args, **kwargs):
        # Compute the padding needed for "same" convolution
        self.kernel_size = (kernel_size,) * 3 if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride,) * 3 if isinstance(stride, int) else stride
        self.dilation = (dilation,) * 3 if isinstance(dilation, int) else dilation

        # Calculate "same" padding for each dimension
        self.padding = tuple(
            (k - 1) // 2 for k in self.kernel_size
        )

        super(Conv3DSame, self).__init__(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(x)

#  In case someone wonders... https://discuss.pytorch.org/t/is-there-any-different-between-torch-sigmoid-and-torch-nn-functional-sigmoid/995
class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)
      
class Sigmoid(nn.Module):
    def forward(self, x):
        return F.sigmoid(x)
    
class Tanh(nn.Module):
    def forward(self, x):
        return F.tanh(x)

def get_activation(activation: str) -> nn.Module:
    if activation is None: return nn.Identity()
    if activation == "relu": return nn.ReLU()
    if activation == "sigmoid": return Sigmoid()