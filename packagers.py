from torch.nn import Module
from torch import stack, empty_like

class Type1EmptyPackager(Module):
    def forward(self, tensor):
        empty_tensor = empty_like(tensor)
        out = [tensor, empty_tensor]
        return stack(out, dim=-1)

class Type2EmptyPackager(Module):
    def forward(self, tensor):
        empty_tensor = empty_like(tensor)
        out = [empty_tensor, tensor]
        return stack(out, dim=-1)

class FuseAsCarryPackager(Module):
    def __init__(self, tensor):
        super().__init__()
        self.carry = tensor

    def forward(self, tensor):
        return stack([tensor, self.carry], dim=-1)

class FuseAsActivatorPackager(Module):
    def __init__(self, tensor):
        super().__init__()
        self.activator = tensor

    def forward(self, tensor):
        return stack([self.activator, tensor], dim=-1)

__all__ = ['Type1EmptyPackager', 'Type2EmptyPackager', 'FuseAsCarryPackager', 'FuseAsActivatorPackager']