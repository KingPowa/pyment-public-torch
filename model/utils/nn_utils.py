import torch.nn as nn
import torch as th
import torch.nn.functional as F

class Conv3DSame(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super(Conv3DSame, self).__init__(*args, **kwargs)
        # Calculate padding
        self.zeropad = nn.ZeroPad3d(
            sum([(k // 2, k // 2 + (k - 1) % 2) for k in self.kernel_size[::-1]], ())
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self._conv_forward(self.zeropad(x), self.weight, self.bias)

#  In case someone wonders... https://discuss.pytorch.org/t/is-there-any-different-between-torch-sigmoid-and-torch-nn-functional-sigmoid/995
class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)
      
class Sigmoid(nn.Module):
    def forward(self, x):
        return F.sigmoid(x)
    
class Relu(nn.Module):
    def forward(self, x):
        return F.relu(x)
    
class Tanh(nn.Module):
    def forward(self, x):
        return F.tanh(x)

def get_activation(activation: str) -> nn.Module:
    if activation is None: return nn.Identity()
    if activation == "relu": return Relu()
    if activation == "sigmoid": return Sigmoid()