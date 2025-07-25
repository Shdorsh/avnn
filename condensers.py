from torch import randn, zeros
from torch.nn import Module, Parameter
from torch.nn.functional import relu

class MappingCondenser(Module):
    def forward(self, avnn_tensor):
        # Input shape: [batch, dim, 2]
        # Output shape: [batch, dim * 2]
        return avnn_tensor.view(avnn_tensor.shape[0], -1)


class FusingCondenser(Module):
    def __init__(self, input_dim, activation_function=relu):
        super().__init__()
        self.x_weight = Parameter(randn(input_dim))
        self.y_weight = Parameter(randn(input_dim))
        self.bias = Parameter(zeros(input_dim))
        self.activation_function = activation_function

    def forward(self, avnn_tensor):
        # Shape: [batch, dim, 2]
        x = avnn_tensor[..., 0]
        y = avnn_tensor[..., 1]
        return self.activation_function(x * self.x_weight + y * self.y_weight + self.bias)

class SeparatingCondenser(Module):
    def forward(self, avnn_tensor):
        # Shape: [batch, dim, 2]
        x = avnn_tensor[..., 0]
        y = avnn_tensor[..., 1]
        return x, y  # two separate [batch, dim] tensors

__all__ = ['MappingCondenser', 'FusingCondenser', 'SeparatingCondenser']