import paddle
import numpy as np
from paddlenlp.transformers import LayoutLMv2Model, LayoutLMv2Tokenizer

def LayerCase():
    """模型库中间态"""
    model = LayoutLMv2Model.from_pretrained('layoutlmv2-base-uncased')
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
    tokenizer = LayoutLMv2Tokenizer.from_pretrained('layoutlmv2-base-uncased')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = (
        paddle.to_tensor(inputs_dict['input_ids'], stop_gradient=False),
        None,
        None,
        paddle.to_tensor(inputs_dict['token_type_ids'], stop_gradient=False),
    )
    return inputs


def create_numpy_inputs():
    tokenizer = LayoutLMv2Tokenizer.from_pretrained('layoutlmv2-base-uncased')
    inputs_dict = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
    inputs = (
        np.array([inputs_dict['input_ids']]),
        None,
        None,
        np.array([inputs_dict['token_type_ids']]),
    )
    return inputs
