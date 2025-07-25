from derives import derived_adjustedmean
from layers import AVNNType1Linear, AVNNType2Linear, AVNNType1Conv2d, AVNNType2Conv2d
from torch.nn import Module, Sequential
from torch.nn.functional import relu

class AVNNLinearBlock(Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=None, derived_mode=derived_adjustedmean,
                 activation=relu):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        output_dim = output_dim or input_dim
        
        self.layers = Sequential(
            AVNNType1Linear(input_dim=input_dim, output_dim=hidden_dim, 
                           derive_mode=derived_mode, activation=activation),
            AVNNType2Linear(input_dim=hidden_dim, output_dim=output_dim, 
                           derive_mode=derived_mode, activation=activation)
        )

    def forward(self, tensor):
        return self.layers(tensor)


class AVNNConv2dBlock(Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, kernel_size=1, stride=1, padding=0, derive_mode=derived_adjustedmean, activation=relu):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.layers = Sequential(
            AVNNType1Conv2d(in_channels=in_channels, out_channels=hidden_channels, stride=stride,
                           derive_mode=derive_mode, activation=activation, kernel_size=kernel_size, padding=padding),
            AVNNType2Conv2d(in_channels=hidden_channels, out_channels=out_channels,
                           derive_mode=derive_mode, activation=activation, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, tensor):
        return self.layers(tensor)


class AVNNResBlock(Module):
    def __init__(self, dim, hidden_dim=None, derived_mode=derived_adjustedmean, activation=relu):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.layers = AVNNLinearBlock(input_dim=dim, hidden_dim=hidden_dim, output_dim=dim, derived_mode=derived_mode, activation=relu)
        self.activation = activation

    def forward(self, x):
        out = self.layers(x)
        out = out + x

class AVNNConv2dResBlock(Module):
    def __init__(self, channels, hidden_channels=None, kernel_size=1, stride=1, padding=0, derive_mode=derived_adjustedmean, activation=relu):
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.layers = AVNNConv2dBlock(in_channels=channels, hidden_channels=hidden_channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding, derive_mode=derive_mode, activation=activation)

    def forward(self, x):
        out = self.layers(x)
        return out + x

__all__ = ['AVNNLinearBlock', 'AVNNConv2dBlock', 'AVNNResBlock', 'AVNNConv2dResBlock']