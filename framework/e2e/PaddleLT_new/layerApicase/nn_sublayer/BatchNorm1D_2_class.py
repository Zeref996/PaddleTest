import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: BatchNorm1D_2
    api简介: 1维BN批归一化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.BatchNorm1D(num_features=1, )

    def forward(self, data, ):
        """
        forward
        """
        out = self.func(data, )
        return out


def create_inputspec():
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec


def create_tensor_inputs():
    """
    numpy array
    """
    inputs = (paddle.to_tensor([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]], dtype='float32', stop_gradient=False), )
    return inputs

def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[[0.6964692, 0.28613934, 0.22685145]], [[0.5513148, 0.71946895, 0.42310646]]]).astype('float32'), )
    return inputs
