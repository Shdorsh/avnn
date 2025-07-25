from torch.nn import Module

class AVNNLinearToConvBridge(Module):
    def __init__(self, channels, width, height):
        super().__init__()
        self.channels, self.width, self.height = channels, width, height

    def forward(self, tensor):
        self.channels = self.channels or tensor.shape[0]
        B, C, H, W, two = tensor.shape
        assert two == 2
        return tensor.view(B, C, H, W, 2)

__all__ = ['AVNNConvToLinearBridge']