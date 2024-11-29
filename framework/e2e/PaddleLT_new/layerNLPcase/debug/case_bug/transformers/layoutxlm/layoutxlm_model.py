import paddle
import numpy as np
from paddlenlp.transformers import LayoutXLMModel, LayoutXLMTokenizer

def LayerCase():
    """模型库中间态"""
    model = LayoutXLMModel.from_pretrained('layoutxlm-base-uncased')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 13), dtype=paddle.float32, stop_gradient=False),
        None,
        None,
        paddle.static.InputSpec(shape=(-1, 13), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = LayoutXLMTokenizer.from_pretrained('layoutxlm-base-uncased')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = (
        paddle.to_tensor(inputs_dict['input_ids'], stop_gradient=False),
        None,
        None,
        paddle.to_tensor(inputs_dict['token_type_ids'], stop_gradient=False),
    )
    return inputs


def create_numpy_inputs():
    tokenizer = LayoutXLMTokenizer.from_pretrained('layoutxlm-base-uncased')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = (
        np.array([inputs_dict['input_ids']]),
        None,
        None,
        np.array([inputs_dict['token_type_ids']]),
    )
    return inputs
