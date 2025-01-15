import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: addmm_zero_size_0_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, data, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.addmm(data, x, y, alpha=5.0, beta=0.5 )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.float32, stop_gradient=False),
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (
        paddle.to_tensor(-10 + (10 - -10) * np.random.random([100, 0]).astype('float32'), dtype='float32', stop_gradient=False), 
        paddle.to_tensor(-10 + (10 - -10) * np.random.random([100, 0]).astype('float32'), dtype='float32', stop_gradient=False), 
        paddle.to_tensor(-10 + (10 - -10) * np.random.random([100, 0]).astype('float32'), dtype='float32', stop_gradient=False), 
    )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (
        -10 + (10 - -10) * np.random.random([100, 0]).astype('float32'), 
        -10 + (10 - -10) * np.random.random([100, 0]).astype('float32'),
        -10 + (10 - -10) * np.random.random([100, 0]).astype('float32'),
    )
    return inputs
