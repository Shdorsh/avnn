from derives import derived_adjustedmean
from helpers import flatten_avnn
from torch import nn, stack, empty_like, empty
from torch.nn import Module, Linear
from torch.nn.functional import relu

class AVNNType1Linear(Module):
    """Type 1: Value (first float) is used as 'activator', Meaning (second float) is used as carry,
    meaning literally just being carried into the next tensor"""
    def __init__(self, input_dim, output_dim, derive_mode=derived_adjustedmean, activation=relu):
        # setting a valide derive_mode
        super().__init__()
        if not callable(derive_mode):
            self.derive_mode = derived_adjustedmean
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
        if input_tensor.ndim > 3:
            input_tensor = flatten_avnn(input_tensor, 3)
        # Separate X and Y channels
        activation_in = input_tensor[..., self.activator]
        carry_in      = input_tensor[..., self.carry]
        
        # Apply linear + activation to the activation channel
        activation_out = self.activation(self.linear(activation_in))

        # Preallocate carry_out tensor of the right shape
        batch_size = activation_out.shape[0]
        carry_out = empty_like(activation_out)

        # Fill each batch entry with the derived scalar
        for i in range(batch_size):
            val = self.derive_mode(activation_in[i], carry_in[i])
            carry_out[i].fill_(val)

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

class AVNNType1Conv2d(nn.Module):
    """Type 1: Value (first float) is used as 'activator', Meaning (second float) is used as carry,
    meaning literally just being carried into the next tensor"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, derive_mode=derived_adjustedmean, activation=relu):
        super().__init__()
        # X-path conv
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # Y-path pooling
        if not callable(derive_mode):
            self.derive_mode = derived_adjustedmean
            raise ValueError("Error: Given derive function not valid! Defaulting to avnn.derives.derived_adjustedmean")
        else:
            self.derive_mode = derive_mode
        # Project pooled Y to match out_channels
        self.y_proj = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)

        # setting a valid activator
        if not callable(activation):
            self.derive_mode = relu
            raise ValueError("Error: Given derive function not valid! Defaulting to torch.nn.functional.relu")
        else:
            self.activation = activation
        self._set_activator_and_carry()
        self.k, self.s, self.p = kernel_size, stride, padding

    def _set_activator_and_carry(self):
        self.activator = 0
        self.carry = 1

    def forward(self, input_tensor):
        if input_tensor.ndim > 5:
            input_tensor = flatten_avnn(input_tensor, 5)
        act_in = input_tensor[..., self.activator]  # [B, C_in, H, W]
        car_in = input_tensor[..., self.carry]      # [B, C_in, H, W]

        # 1) Activator path: conv + activation
        act_out = self.activation(self.conv_x(act_in))  # [B, C_out, H', W']

        # 2) Carry path: patch‐wise derive on BOTH act_in *and* car_in
        patches_x = act_in.unfold(2, self.k, self.s).unfold(3, self.k, self.s)  # [B,C,H',W',k,k]
        patches_y = car_in.unfold(2, self.k, self.s).unfold(3, self.k, self.s)

        B, C, Hp, Wp, _, _ = patches_x.shape
        patches_x = patches_x.contiguous().view(B, C, Hp, Wp, -1)
        patches_y = patches_y.contiguous().view(B, C, Hp, Wp, -1)

        # allocate carry_out
        carry_out = empty((B, C, Hp, Wp), device=input_tensor.device, dtype=act_out.dtype)

        # for each position, derive from the local X/Y lists
        for b in range(B):
            for c in range(C):
                for i in range(Hp):
                    for j in range(Wp):
                        x_list = patches_x[b, c, i, j]
                        y_list = patches_y[b, c, i, j]
                        carry_out[b, c, i, j] = self.derive_mode(x_list, y_list)

        # 3) Project carry_out → C_out channels (if needed)
        carry_out = self.y_proj(carry_out)  # [B, C_out, H', W']

        # 4) Pack back into AVNN
        result = [None, None]
        result[self.activator] = act_out
        result[self.carry]     = carry_out
        return stack(result, dim=-1)  # [B, C_out, H', W', 2]


class AVNNType2Conv2d(AVNNType1Conv2d):
    """Type 2: Value (first float) is used as 'carry', Meaning (second float) is used as 'activator'"""
    def _set_activator_and_carry(self):
        self.activator = 1
        self.carry = 0

__all__ = ['AVNNType1Linear', 'AVNNType2Linear', 'AVNNType1Conv2d', 'AVNNType2Conv2d']