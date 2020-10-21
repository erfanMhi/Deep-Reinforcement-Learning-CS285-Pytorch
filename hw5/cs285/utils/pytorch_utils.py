import torch
import torch.nn as nn
import functools
import numpy as np
import inspect

import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, output_size, n_layers, 
            size, activation=torch.tanh, output_activation=None):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.size = size
        self.n_layers = n_layers
        self.output_activation = output_activation
        
        layers_size = [self.input_size] + ([self.size]*self.n_layers) + [self.output_size]
        self.layers = nn.ModuleList([nn.Linear(layers_size[i], layers_size[i+1]) 
                                    for i in range(len(layers_size)-1)])
        
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        

    def forward(self, x):
        
        out = x
        for i, layer in enumerate(self.layers):
            if i!=len(self.layers)-1:
                out = self.activation(layer(out))
            else:
                out = layer(out)

        if self.output_activation is not None:
            out = self.output_activation(out)
       
        return out


def multivariate_normal_diag(loc, scale_diag):
    normal = torch.distributions.Normal(loc, scale=scale_diag)
    return torch.distributions.Independent(normal, 1)

def _is_method(func):
    spec = inspect.signature(func)
    return 'self' in spec.parameters

def convert_args_to_tensor(positional_args_list=None, keyword_args_list=None, device='cpu'):
    """A decorator which converts args in positional_args_list to torch.Tensor

    Args:
        positional_args_list ([list]): [arguments to be converted to torch.Tensor. If None, 
        it will convert all positional arguments to Tensor]
        keyword_args_list ([list]): [arguments to be converted to torch.Tensor. If None, 
        it will convert all keyword arguments to Tensor]
        device ([str]): [pytorch will run on this device]
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            
            
            
            _device = device
            _keyword_args_list = keyword_args_list
            _positional_args_list = positional_args_list
            
            if keyword_args_list is None:
                _keyword_args_list = list(kwargs.keys())

            if positional_args_list is None:
                _positional_args_list = list(range(len(args)))
            
                if _is_method(func):
                    _positional_args_list = _positional_args_list[1:]
            
            args = list(args)
            for i, arg in enumerate(args):
                if i in _positional_args_list:
                    if type(arg) == np.ndarray:
                        if arg.dtype == np.double:
                            args[i] = torch.from_numpy(arg).type(torch.float32).to(_device)
                        else:
                            args[i] = torch.from_numpy(arg).to(_device)
                    elif type(arg) == list:
                        args[i] = torch.tensor(arg).to(_device)
                    elif type(arg) == torch.Tensor or type(arg) == int or type(arg) == float or type(arg) == bool:
                        continue
                    else:
                        raise ValueError('Arguments should be Numpy arrays, but argument in position {} is not: {}'.format(str(i), type(arg)))
            
            for key, arg in kwargs.items():
                if key in _keyword_args_list:
                    if type(arg) == np.ndarray:
                        if arg.dtype == np.double:
                            kwargs[key] = torch.from_numpy(arg).type(torch.float32).to(_device)
                        else:
                            kwargs[key] = torch.from_numpy(arg).to(_device) 
                    elif type(arg) == list:
                        kwargs[key] = torch.tensor(arg).to(_device)
                    elif type(arg) == torch.Tensor or type(arg) == int or type(arg) == float or type(arg) == bool:
                        continue
                    else:
                        raise ValueError('Arguments should be Numpy arrays, but argument in position {} is not: {}'.format(str(i), type(arg)))

            return func(*args, **kwargs)

        return wrapper

    return decorator

@convert_args_to_tensor([0], ['labels'])
def torch_one_hot(labels, one_hot_size):
    one_hot = torch.zeros(labels.shape[0], one_hot_size, device=labels.device)
    one_hot[torch.arange(labels.shape[0], device=labels.device), labels] = 1
    return one_hot

@convert_args_to_tensor()
def gather_nd(params, indices):
    """params is of "n" dimensions and has size [x1, x2, x3, ..., xn], indices is of 2 dimensions  and has size [num_samples, m] (m <= n)"""
    assert type(indices) == torch.Tensor
    return params[indices.transpose(0,1).long().numpy().tolist()]
