import torch
import torch.nn as nn
import functools
import numpy as np
import inspect


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

def convert_args_to_tensor(positional_args_list=None, keyword_args_list=None):
    """A decorator which converts args in positional_args_list to torch.Tensor

    Args:
        positional_args_list ([list]): [arguments to be converted to torch.Tensor. If None, 
        it will convert all positional arguments to Tensor]
        keyword_args_list ([list]): [arguments to be converted to torch.Tensor. If None, 
        it will convert all keyword arguments to Tensor]
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            
            
            
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
                        args[i] = torch.from_numpy(arg).type(torch.FloatTensor)
                    elif type(arg) == torch.Tensor:
                        pass
                    else:
                        raise ValueError('Arguments should be Numpy arrays, but argument in position {} is not'.format(str(i)))
            
            for key, arg in kwargs.items():
                if key in _keyword_args_list:
                    if type(arg) == np.ndarray:
                        kwargs[key] = torch.from_numpy(arg).type(torch.FloatTensor)
                    elif type(arg) == torch.Tensor:
                        pass
                    else:
                        raise ValueError('Arguments should be Numpy arrays, but argument {} is not'.format(str(key)))
            
            return func(*args, **kwargs)

        return wrapper

    return decorator