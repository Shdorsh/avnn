from torch.nn import Module

class AVNNConvToLinearBridge(Module):
    def forward(self, tensor):
        B, C, H, W, two = tensor.shape
        assert two == 2
        return tensor.view(B, C * H * W, 2)

class AVNNLinearToConvBridge(Module):
    def __init__(self, b, c, h, w):
        super().__init__()
        self.b, self.c, self.h, self.w = b, c, h, w

    def forward(self, tensor):
        B, C, H, W, two = tensor.shape
        assert two == 2
        return tensor.view(B, C, H, W, 2)

__all__ = ['AVNNConvToLinearBridge']