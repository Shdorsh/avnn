from torch import sum as t_sum, tensor
import torch.nn.functional as F

class AVNNDerive:
    def __init__(self):
        raise SystemError("AVNN derive classes are not meant to be instantiated directly, but used as utilities for AVNN layers.")

    @staticmethod
    def scalar(x_vals, y_vals, temperature=1.0):
        raise NotImplementedError("Override scalar mode.")

    @staticmethod
    def batch(x_vals, y_vals, temperature=1.0):
        raise NotImplementedError("Override batch mode.")

class AVNNDeriveMax(AVNNDerive):
    """Soft approximation of selecting y corresponding to max(x)"""
    @staticmethod
    def scalar(x_vals, y_vals, temperature=1.0):
        weights = F.softmax(x_vals / temperature, dim=0)
        return t_sum(weights * y_vals)

    @staticmethod
    def batch(x_vals, y_vals, temperature=1.0):
        weights = F.softmax(x_vals / temperature, dim=-1)
        return t_sum(weights * y_vals, dim=-1)


class AVNNDeriveMin(AVNNDerive):
    """Soft approximation of selecting y corresponding to min(x)"""
    @staticmethod
    def scalar(x_vals, y_vals, temperature=1.0):
        weights = F.softmax(-x_vals / temperature, dim=0)
        return t_sum(weights * y_vals)

    @staticmethod
    def batch(x_vals, y_vals, temperature=1.0):
        weights = F.softmax(-x_vals / temperature, dim=-1)
        return t_sum(weights * y_vals, dim=-1)


class AVNNDeriveAdjustedMin(AVNNDerive):
    """Soft min over positive x_vals only"""
    @staticmethod
    def scalar(x_vals, y_vals, temperature=1.0):
        mask = (x_vals > 0).float()
        if mask.sum() == 0:
            return tensor(0.0, device=x_vals.device)
        masked_x = x_vals * mask + (1 - mask) * (-1e9)
        weights = F.softmax(-masked_x / temperature, dim=0)
        return t_sum(weights * y_vals)

    @staticmethod
    def batch(x_vals, y_vals, temperature=1.0):
        mask = (x_vals > 0).float()
        masked_x = x_vals * mask + (1 - mask) * (-1e9)
        weights = F.softmax(-masked_x / temperature, dim=-1)
        return t_sum(weights * y_vals, dim=-1)


class AVNNDeriveMean(AVNNDerive):
    """Soft average using attention-style weighting (meanish)"""
    @staticmethod
    def scalar(x_vals, y_vals, temperature=1.0):
        weights = F.softmax(x_vals / temperature, dim=0)
        return t_sum(weights * y_vals)

    @staticmethod
    def batch(x_vals, y_vals, temperature=1.0):
        weights = F.softmax(x_vals / temperature, dim=-1)
        return t_sum(weights * y_vals, dim=-1)


class AVNNDeriveAdjustedMean(AVNNDerive):
    """Soft average over positive x_vals only"""
    @staticmethod
    def scalar(x_vals, y_vals, temperature=1.0):
        mask = (x_vals > 0).float()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=x_vals.device)
        masked_x = x_vals * mask + (1 - mask) * (-1e9)
        weights = F.softmax(masked_x / temperature, dim=0)
        return t_sum(weights * y_vals)

    @staticmethod
    def batch(x_vals, y_vals, temperature=1.0):
        mask = (x_vals > 0).float()
        masked_x = x_vals * mask + (1 - mask) * (-1e9)
        weights = F.softmax(masked_x / temperature, dim=-1)
        return t_sum(weights * y_vals, dim=-1)

__all__ = ['AVNNDeriveMax', 'AVNNDeriveMin', 'AVNNDeriveAdjustedMin', 'AVNNDeriveMean', 'AVNNDeriveAdjustedMean']