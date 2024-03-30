from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

class MLP(nn.Module):
    def __init__(self, input_size, size, output_size, n_layers, activation, output_activation):
        super().__init__()

        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        
        
        self.linear1 = nn.Linear(input_size, size)
        self.linear2 = nn.Linear(size, size)
        self.linear3 = nn.Linear(size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        for _ in range(self.n_layers - 1):
            x = self.linear2(x)
            x = self.activation(x)

        output = self.linear3(x)

        classes = output[:4]
        classes = nn.functional.sigmoid(classes)

        regressor = output[4]
        regressor = nn.functional.relu(regressor)
        return classes, regressor
    
def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    mlp = MLP(input_size, size, output_size, n_layers, activation, output_activation)
    
    return mlp


    
class task_MLP(nn.Module):
    def __init__(self, input_size, size, output_size, n_layers, activation, output_activation):
        super().__init__()

        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
        
        
        self.linear1 = nn.Linear(input_size, size)
        self.linear2 = nn.Linear(size, size)
        self.linear3 = nn.Linear(size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        for _ in range(self.n_layers - 1):
            x = self.linear2(x)
            x = self.activation(x)

        x = self.linear3(x)
        x = nn.functional.sigmoid(x)
        return x

class regress_MLP(nn.Module):
    def __init__(self, input_size, size, output_size, n_layers, activation, output_activation):
        super().__init__()

        self.n_layers = n_layers
        self.size = size
        self.activation = activation
        self.output_activation = output_activation
          
        self.linear1 = nn.Linear(input_size, size)
        self.linear2 = nn.Linear(size, size)
        self.linear3 = nn.Linear(size, output_size)
        self.activation = nn.ReLU()


    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        for _ in range(self.n_layers - 1):
            x = self.linear2(x)
            x = self.activation(x)

        x = self.linear3(x)
        x = nn.functional.relu(x)
        
        return x
device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
