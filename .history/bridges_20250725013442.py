from torch import cat
from torch.nn import Module

class AVNNConv2dToLinarBridge(Module):
    def forward(self, tensor):
        B, C, H, W, two = tensor.shape
        return tensor.view(B, C * H * W, two)

class AVNNLinearToConv2dBridge(Module):
    def __init__(self, channels, height, width):
        super().__init__()
        self.C = channels
        self.H = height
        self.W = width

    def forward(self, tensor):  # tensor shape: [B, F, 2]
        B, F, two = tensor.shape
        target_dim = self.C * self.H * self.W
        if F > target_dim:
            raise ValueError(f"Feature dim {F} too big for target shape {target_dim}")
        if F < target_dim:
            # pad with zeros (or noise)
            pad_size = target_dim - F
            pad_tensor = tensor.new_zeros(B, pad_size, two)
            tensor = cat([tensor, pad_tensor], dim=1)
        # reshape to conv format
        return tensor.view(B, self.C, self.H, self.W, two)

__all__ = ['AVNNConv2dToLinarBridge', 'AVNNLinearToConv2dBridge']