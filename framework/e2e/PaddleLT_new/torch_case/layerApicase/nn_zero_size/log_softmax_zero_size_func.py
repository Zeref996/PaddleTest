import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: log_softmax_zero_size_class
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, data):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = nn.functional.log_softmax(data, axis=-1)
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = (torch.tensor((-2 + 7 * np.random.random([3, 0, 1, 1])).astype(np.float32), dtype=torch.float32, requires_grad=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ((-2 + 7 * np.random.random([3, 0, 1, 1])).astype('float32'),)
    return inputs