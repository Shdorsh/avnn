import torch
import torch.nn.functional as F

def derived_max(x_vals, y_vals, temperature=1.0):
    """
    Soft approximation of selecting y corresponding to max(x)
    """
    weights = F.softmax(x_vals / temperature, dim=0)
    return torch.sum(weights * y_vals)

def derived_min(x_vals, y_vals, temperature=1.0):
    """
    Soft approximation of selecting y corresponding to min(x)
    """
    weights = F.softmax(-x_vals / temperature, dim=0)
    return torch.sum(weights * y_vals)

def derived_adjustedmin(x_vals, y_vals, temperature=1.0):
    """
    Soft min over positive x_vals only
    """
    mask = (x_vals > 0).float()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x_vals.device)
    # Apply a large negative bias to masked-out elements
    masked_x = x_vals * mask + (1 - mask) * (-1e9)
    weights = F.softmax(-masked_x / temperature, dim=0)
    return torch.sum(weights * y_vals)

def derived_mean(x_vals, y_vals, temperature=1.0):
    """
    Soft median approximation: use softmax over x (sorted-like effect)
    """
    weights = F.softmax(x_vals / temperature, dim=0)
    return torch.sum(weights * y_vals)

def derived_adjustedmean(x_vals, y_vals, temperature=1.0):
    """
    Soft median over positive x_vals only
    """
    mask = (x_vals > 0).float()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x_vals.device)
    masked_x = x_vals * mask + (1 - mask) * (-1e9)
    weights = F.softmax(masked_x / temperature, dim=0)
    return torch.sum(weights * y_vals)

__all__ = ['derived_max', 'derived_min', 'derived_adjustedmin', 'derived_mean', 'derived_adjustedmean']