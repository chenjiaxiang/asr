from typing import Tuple
import torch.nn as nn 
import torch.nn.init as init
from torch import Tensor

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_normal_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class View(nn.Module):
    def __int__(self, shape: Tuple, contiguous: bool = False) -> None:
        super(View, self).__init__()
        self.shape = shape
        self.contiguous = contiguous

    def forward(self, inputs: Tensor):
        if self.contiguous:
            inputs = inputs.contiguous()
        return inputs.view(*self.shape)
    
class Transpose(nn.Module):
    def __init__(self, shape: Tuple) -> None:
        super(Transpose, self).__init__()
        self.shape = shape
        
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.transpose(*self.shape)
