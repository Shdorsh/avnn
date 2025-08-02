from derives import AVNNDeriveAdjustedMean
from torch import stack
from torch.nn import Module, Linear, Conv2d
from torch.nn.functional import relu

class AVNNType1Linear(Module):
    """Type 1: Value (first float) is used as 'activator', Meaning (second float) is used as carry,
    meaning literally just being carried into the next tensor"""
    def __init__(self, input_dim, output_dim, derive_mode=AVNNDeriveAdjustedMean, activation=relu):
        # setting a valide derive_mode
        super().__init__()
        if not callable(derive_mode):
            self.derive_mode = AVNNDeriveAdjustedMean
            raise ValueError("Error: Given derive function not valid! Defaulting to avnn.derives.derived_adjustedmean")
        else:
            self.derive_mode = derive_mode

        # setting a valid activator
        if not callable(activation):
            self.activation = relu
            raise ValueError("Error: Given derive function not valid! Defaulting to torch.nn.functional.relu")
        else:
            self.activation = activation
        self.linear = Linear(input_dim, output_dim)
        self._set_activator_and_carry()

    def _set_activator_and_carry(self):
        self.activator = 0
        self.carry = 1

    def forward(self, input_tensor):
        # Separate X and Y channels
        activation_in = input_tensor[..., self.activator]
        carry_in      = input_tensor[..., self.carry]
        
        # Apply linear + activation to the activation channel
        activation_out = self.activation(self.linear(activation_in))

        # Fill each batch entry with the derived scalar
        val = self.derive_mode.batch(activation_in, carry_in)
        carry_out = val.unsqueeze(-1).expand_as(activation_out)

        # Pack back into an AVNN tensor
        out = [None, None]
        out[self.activator] = activation_out
        out[self.carry]     = carry_out
        return stack(out, dim=-1)


class AVNNType2Linear(AVNNType1Linear):
    """Type 2: Value (first float) is used as 'carry', Meaning (second float) is used as 'activator'"""
    def _set_activator_and_carry(self):
        self.activator = 1
        self.carry = 0

class AVNNType1Conv2d(Module):
    """
    Type 1: Value (first float) is used as 'activator', Meaning (second float) is used as carry.
    Meaning is carried forward using a soft derive function over patch neighborhoods.
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 derive_mode=AVNNDeriveAdjustedMean, activation=relu):
        super().__init__()

        self.k, self.s, self.p = kernel_size, stride, padding

        # Activator (value) path: standard Conv2d + nonlinearity
        self.conv_x = Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        # Carry (meaning) projection path after derivation
        self.y_proj = Conv2d(in_channels, out_channels, kernel_size=1)

        # Check and assign derive function
        if not callable(derive_mode) or not hasattr(derive_mode, 'batch'):
            self.derive_mode = AVNNDeriveAdjustedMean
            raise ValueError("Given derive function is not valid. Defaulting to AVNNDeriveAdjustedMean.")
        else:
            self.derive_mode = derive_mode

        # Check and assign activation function
        if not callable(activation):
            self.activation = relu
            raise ValueError("Given activation is not valid. Defaulting to ReLU.")
        else:
            self.activation = activation

        self._set_activator_and_carry()

    def _set_activator_and_carry(self):
        self.activator = 0
        self.carry = 1

    def forward(self, input_tensor):
        # Separate input into activator (value) and carry (meaning) parts
        act_in = input_tensor[..., self.activator]  # [B, C_in, H, W]
        car_in = input_tensor[..., self.carry]      # [B, C_in, H, W]

        # 1) Activator path: convolution + activation
        act_out = self.activation(self.conv_x(act_in))  # [B, C_out, H', W']

        # 2) Carry path: unfold into patches, derive carry from patchwise info
        patches_x = act_in.unfold(2, self.k, self.s).unfold(3, self.k, self.s)  # [B, C, Hp, Wp, k, k]
        patches_y = car_in.unfold(2, self.k, self.s).unfold(3, self.k, self.s)

        B, C, Hp, Wp, _, _ = patches_x.shape
        patches_x = patches_x.contiguous().view(B, C, Hp, Wp, -1)  # [B, C, Hp, Wp, K*K]
        patches_y = patches_y.contiguous().view(B, C, Hp, Wp, -1)

        # 3) Vectorized carry derivation
        carry_out = self.derive_mode.batch(patches_x, patches_y)  # [B, C, Hp, Wp]

        # 4) Project to match out_channels
        carry_out = self.y_proj(carry_out)  # [B, C_out, H', W']

        # 5) Reconstruct AVNN tensor: [B, C_out, H', W', 2]
        result = [None, None]
        result[self.activator] = act_out
        result[self.carry]     = carry_out
        return stack(result, dim=-1)


class AVNNType2Conv2d(AVNNType1Conv2d):
    """Type 2: Value (first float) is used as 'carry', Meaning (second float) is used as 'activator'"""
    def _set_activator_and_carry(self):
        self.activator = 1
        self.carry = 0

__all__ = ['AVNNType1Linear', 'AVNNType2Linear', 'AVNNType1Conv2d', 'AVNNType2Conv2d']