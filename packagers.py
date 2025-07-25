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

class FuseAsMeaningPackager(Module):
    def __init__(self, tensor):
        super().__init__()
        self.meaning = tensor

    def forward(self, tensor):
        return stack([tensor, self.meanin], dim=-1)

class FuseAsValuePackager(Module):
    def __init__(self, tensor):
        super().__init__()
        self.value = tensor

    def forward(self, tensor):
        return stack([self.value, tensor], dim=-1)

__all__ = ['Type1EmptyPackager', 'Type2EmptyPackager', 'FuseAsCarryPackager', 'FuseAsActivatorPackager']