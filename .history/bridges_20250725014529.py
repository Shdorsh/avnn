from torch import cat
from torch.nn import Module, Linear

class AVNNConv2dToLinarBridge(Module):
    def __init__(self, dim, channels, height, width):
        super().__init__()
        self.dim = dim
        self.features = channels * height * width

    def forward(self, tensor):
        B, _, _, _, two = tensor.shape
        tensor = tensor.view(B, self.features, two)
        linear = Linear(in_features=self.features, out_features=self.dim)
        return linear(tensor)

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