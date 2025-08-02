from torch import cat, stack
from torch.nn import Module, Linear

class AVNNConv2dToLinearBridge(Module):
    def __init__(self, dim, channels, height, width):
        super().__init__()
        self.dim = dim
        self.features = channels * height * width
        self.linear_x = Linear(in_features=self.features, out_features=self.dim)
        self.linear_y = Linear(in_features=self.features, out_features=self.dim)

    def forward(self, tensor):
        B, _, _, _, two = tensor.shape
        tensor = tensor.view(B, self.features, two)
        tensor_x = self.linear_x(tensor[..., 0])
        tensor_y = self.linear_y(tensor[..., 1])
        return stack([tensor_x, tensor_y], dim=-1)

class AVNNLinearToConv2dBridge(Module):
    def __init__(self, channels, height, width, embedding_dim=1):
        super().__init__()
        self.C = channels
        self.H = height
        self.W = width
    
    def _unembedded_forward(self, tensor):
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

    def _embedded_forward(self, tensor):
        B, F, embedding_dim, two = tensor.shape
        if F != self.C * self.H * self.W:
            raise ValueError(f"Feature dim {F} does not match expected shape {self.C * self.H * self.W}")
        if embedding_dim != 2:
            raise ValueError(f"Expected embedding_dim of 2, got {embedding_dim}")
        return tensor.view(B, self.C, self.H, self.W, two)

    def forward(self, tensor):  # tensor shape: [B, F, 2] or [B, F, embedding_dim, 2]
        print("Tensor dimension: ", tensor.ndim)
        print("Tensor shape: ", tensor.shape)
        if tensor.ndim < 3 or tensor.shape[-1] != 2:
            raise ValueError(f"Expected AVNN tensor with last dimension of size 2, got {tensor.shape}")
        if tensor.ndim == 3:
            return self._unembedded_forward(tensor)
        elif tensor.ndim == 4:
            return self._embedded_forward(tensor)
        raise ValueError(f"Unsupported tensor shape {tensor.shape} for AVNNLinearToConv2dBridge")

class AVNNEmbeddedToConv2dBridge(Module):
    def __init__(self, F, E, C, H, W):
        super().__init__()
        self.F, self.E = F, E
        self.C, self.H, self.W = C, H, W
        self.fc_v = Linear(in_features=F * E, out_features=C * H * W)
        self.fc_m = Linear(in_features=F * E, out_features=C * H * W)

    def forward(self, x):
        print("Input shape:", x.shape)
        print(f"Expected shape: [B, F, E, 2] where F={self.F}, E={self.E}, C={self.C}, H={self.H}, W={self.W}")
        B, F, E, _two = x.shape
        # Starting from [B, F, E, 2]
        x = x.reshape(B, F * E, 2)                          # [B, FE, 2]
        values = x[..., 0]                                  # [B, FE]
        meanings = x[..., 1]                                # [B, FE]

        # Two parallel 1×1 convolutions:
        v_map = self.fc_v(values)                           # [B, C*H*W]
        m_map = self.fc_m(meanings)                         # [B, C*H*W]

        v_map = v_map.view(B, self.C, self.H, self.W)       # [B, C, H, W]
        m_map = m_map.view(B, self.C, self.H, self.W)       # [B, C, H, W]

        out = stack([v_map, m_map], dim=-1)                 # [B, C, H, W, 2]
        return out

__all__ = ['AVNNConv2dToLinearBridge', 'AVNNLinearToConv2dBridge', 'AVNNEmbeddedToConv2dBridge']