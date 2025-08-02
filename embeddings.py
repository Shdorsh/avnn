from torch import stack
from torch.nn import Module, Embedding

class AVNNSharedEmbedding(Module):
    def __init__(self, num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight = None,
        _freeze: bool = False,
        device = None,
        dtype = None):
        super().__init__()
        self.embedding = Embedding(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse, _weight, _freeze, device=device, dtype=dtype)
    
    def forward(self, avnn_tensor):
        # Assuming avnn_tensor is of shape [B, F, 2] and we want to use the first feature
        # for embedding lookup.
        values = self.embedding(avnn_tensor[:, :, 0])
        meanings = self.embedding(avnn_tensor[:, :, 1])
        return stack([values, meanings], dim=-1)  # [B, F, sum(embedding_dim), 2]
