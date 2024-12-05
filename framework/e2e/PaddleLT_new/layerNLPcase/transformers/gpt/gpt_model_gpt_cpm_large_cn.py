import paddle
import numpy as np
from paddlenlp.transformers import GPTModel, GPTTokenizer

def LayerCase():
    """模型库中间态"""
    model = GPTModel.from_pretrained('gpt-cpm-large-cn')
    return model

def create_inputspec():
    inputspec = (
        paddle.static.InputSpec(shape=(-1, 13), dtype=paddle.float32, stop_gradient=False),
        )
    return inputspec


def create_tensor_inputs():
    tokenizer = GPTTokenizer.from_pretrained('gpt-cpm-large-cn')
    inputs_dict = tokenizer("欢迎使用百度飞桨！", return_token_type_ids=False)
    inputs = tuple(paddle.to_tensor([v], stop_gradient=False) for (k, v) in inputs_dict.items())
    return inputs


def create_numpy_inputs():
    tokenizer = GPTTokenizer.from_pretrained('gpt-cpm-large-cn')
    inputs_dict = tokenizer("欢迎使用百度飞桨！", return_token_type_ids=False)
    inputs = tuple(np.array([v]) for (k, v) in inputs_dict.items())
    print("inputs.shape",inputs[0].shape)
    return inputs
