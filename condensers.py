from torch import randn, zeros
from torch.nn import Module, Parameter
from torch.nn.functional import relu

class _Condenser(Module):
    def evaluate(self, avnn_tensor):
        return avnn_tensor

    def forward(self, avnn_tensor):
        if  avnn_tensor.ndim < 3 and avnn_tensor.shape[-1] != 2:
            raise ValueError(f"Error: Non-AVNN tensor of shape {avnn_tensor.shape} is not appropriate for MappingCondenser!")
        else:
            return self.evaluate(avnn_tensor)

class MappingCondenser(_Condenser):
    def evaluate(self, avnn_tensor):
        return avnn_tensor.view(avnn_tensor.shape[0], -1)

class FusingCondenser(_Condenser):
    def __init__(self, input_dim, activation_function=relu):
        super().__init__()
        self.x_weight = Parameter(randn(input_dim))
        self.y_weight = Parameter(randn(input_dim))
        self.bias = Parameter(zeros(input_dim))
        self.activation_function = activation_function

    def evaluate(self, avnn_tensor):
        # Shape: [batch, dim, 2]
        x = avnn_tensor[..., 0]
        y = avnn_tensor[..., 1]
        return self.activation_function(x * self.x_weight + y * self.y_weight + self.bias)

class Conv2DFusingCondenser(_Condenser):
    def __init__(self, in_channels, activation_function=relu):
        super().__init__()
        self.x_weight = Parameter(randn(1, in_channels, 1, 1))  # shape-safe for [B, C, H, W]
        self.y_weight = Parameter(randn(1, in_channels, 1, 1))
        self.bias = Parameter(zeros(1, in_channels, 1, 1))
        self.activation_function = activation_function

    def forward(self, x):  # x: [B, C, H, W, 2]
        v = x[..., 0]  # [B, C, H, W]
        m = x[..., 1]  # [B, C, H, W]
        return self.activation_function(v * self.x_weight + m * self.y_weight + self.bias)

class SeparatingCondenser(_Condenser):
    def evaluate(self, avnn_tensor):
        # Shape: [batch, dim, 2]
        x = avnn_tensor[..., 0]
        y = avnn_tensor[..., 1]
        return x, y  # two separate [batch, dim] tensors

__all__ = ['MappingCondenser', 'FusingCondenser', 'SeparatingCondenser', 'Conv2DFusingCondenser']