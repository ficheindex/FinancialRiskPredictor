# -*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn

from model.stg_lib.utils import get_batcnnorm, get_dropout, get_activation

__all__ = [
    'LinearLayer', 'MLPLayer', 'FeatureSelector',
]

class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size()) 
        self.sigma = sigma
        self.device = device
    
    def forward(self, prev_x):
        z = self.mu + self.sigma*self.noise.normal_()*self.training 
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x
    
    def hard_sigmoid(self, x):
        return torch.clamp(x+0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self


class GatingLayer(nn.Module):
    '''To implement L1-based gating layer (so that we can compare L1 with L0(STG) in a fair way)
    '''
    def __init__(self, input_dim, device):
        super(GatingLayer, self).__init__()
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.device = device
    
    def forward(self, prev_x):
        new_x = prev_x * self.mu 
        return new_x
    
    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return torch.sum(torch.abs(x))


class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, batch_norm=None, dropout=None, bias=None, activation=None):
        if bias is None:
            bias = (batch_norm is None)

        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if batch_norm is not None and batch_norm is not False:
            modules.append(get_batcnnorm(batch_norm, out_features, 1))
        if dropout is not None and dropout is not False:
            modules.append(get_dropout(dropout, 1))
        if activation is not None and activation is not False:
            modules.append(get_activation(activation))
        super().__init__(*modules)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
      