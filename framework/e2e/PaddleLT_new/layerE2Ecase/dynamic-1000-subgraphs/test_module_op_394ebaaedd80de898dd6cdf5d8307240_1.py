import os
os.environ['FLAGS_cinn_new_group_scheduler'] = '1'
os.environ['FLAGS_group_schedule_tiling_first'] = '1'
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_cinn_bucket_compile'] = '1'
import sys
import unittest
import numpy as np
from dataclasses import dataclass
import typing as t

@dataclass
class Stage:
    name: str
    env_vars: t.Dict[str, str]

cinn_stages = [
    Stage(
        name="dynamic_to_static",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=False,
            FLAGS_prim_enable_dynamic=False,
        ),
    ),
    Stage(
        name="prim",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
        ),
    ),
    Stage(
        name="infer_symbolic",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=False,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=False,
            FLAGS_check_infer_symbolic=True,
        ),
    ),
	Stage(
        name="frontend",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=True,
            FLAGS_check_infer_symbolic=False,
            FLAGS_enable_fusion_fallback=True,
        ), 
    ),
    Stage(
        name="backend",
        env_vars=dict(
            PADDLE_DEBUG_ENABLE_CINN=True,
            FLAGS_prim_all=True,
            FLAGS_prim_enable_dynamic=True,
            FLAGS_use_cinn=True,
            FLAGS_check_infer_symbolic=False,
            FLAGS_enable_fusion_fallback=False,
        ), 
    ),
]

def GetCinnStageByName(name):
    for stage in cinn_stages:
        if stage.name == name:
            return stage
    return None

def GetCurrentCinnStage():
    name = os.getenv('PADDLE_DEBUG_CINN_STAGE_NAME')
    if name is None:
        return None
    stage_names = [stage.name for stage in cinn_stages]
    assert name in stage_names, (
        f"PADDLE_DEBUG_CINN_STAGE_NAME should be in {stage_names}"
    )
    return GetCinnStageByName(name)

def GetPrevCinnStage(stage):
    for i in range(1, len(cinn_stages)):
        if stage is cinn_stages[i]:
            return cinn_stages[i - 1]
    return None

def IsCinnStageEnableDiff():
    value = os.getenv('PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF')
    enabled = value in {
        '1',
        'true',
        'True',
    }
    if enabled:
        assert GetCurrentCinnStage() is not None
    return enabled

def GetExitCodeAndStdErr(cmd, env):
    env = {
        k:v
        for k, v in env.items()
        if v is not None
    }
    import subprocess
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    return result.returncode, result.stderr

def GetStageExitCodeAndStdErr(stage):
    return GetExitCodeAndStdErr(
        [sys.executable, __file__],
        env=dict(
            PADDLE_DEBUG_CINN_STAGE_NAME=stage.name,
            PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF='0',
            PYTHONPATH=os.getenv('PYTHONPATH'),
            ATHENA_ENABLE_TRY_RUN="False",
        ),
    )

def AthenaTryRunEnabled():
    return os.getenv('ATHENA_ENABLE_TRY_RUN') not in {
        "0",
        "False",
        "false",
        "OFF"
    }

def GetNeedSkipAndSkipMessage():
    current_stage = GetCurrentCinnStage()
    assert current_stage is not None
    if not IsCinnStageEnableDiff():
        return False, ""
    last_stage = GetPrevCinnStage(current_stage)
    if last_stage is None:
        return False, ""
    exitcode, stderr = GetStageExitCodeAndStdErr(last_stage)
    if exitcode != 0:
        return True, f"last stage failed."
    return False, ""

def GetCurrentStageTryRunExitCodeAndStdErr():
    if not AthenaTryRunEnabled():
        return False, ""
    current_stage = GetCurrentCinnStage()
    assert current_stage is not None
    return GetStageExitCodeAndStdErr(current_stage)

def SetDefaultEnv(**env_var2value):
    for env_var, value in env_var2value.items():
        if os.getenv(env_var) is None:
            os.environ[env_var] = str(value)

SetDefaultEnv(
    PADDLE_DEBUG_CINN_STAGE_NAME="backend",
    PADDLE_DEBUG_CINN_STAGE_ENABLE_DIFF=False,
    PADDLE_DEBUG_ENABLE_CINN=True,
    FLAGS_enable_pir_api=True,
    FLAGS_prim_all=True,
    FLAGS_prim_enable_dynamic=True,
    FLAGS_use_cinn=False,
    FLAGS_check_infer_symbolic=False,
    FLAGS_enable_fusion_fallback=False,
)

need_skip, skip_message = GetNeedSkipAndSkipMessage()
try_run_exit_code, try_run_stderr = GetCurrentStageTryRunExitCodeAndStdErr()
class TestTryRun(unittest.TestCase):
    def test_panic(self):
        if not AthenaTryRunEnabled():
            return
        if try_run_exit_code == 0:
            # All unittest cases passed.
            return
        if try_run_exit_code > 0:
            # program failed but not panic.
            return
        # program panicked.
        kOutputLimit = 65536
        message = try_run_stderr[-kOutputLimit:]
        raise RuntimeError(f"panicked. last {kOutputLimit} characters of stderr: \n{message}")

import paddle

def SetEnvVar(env_var2value):
    for env_var, value in env_var2value.items():
        os.environ[env_var] = str(value)
    paddle.set_flags({
        env_var:value
        for env_var, value in env_var2value.items()
        if env_var.startswith('FLAGS_')
    })

if GetCurrentCinnStage() is not None:
    SetEnvVar(GetCurrentCinnStage().env_vars)

def NumOperationsInBlock(block_idx):
    return [31][block_idx] - 1 # number-of-ops-in-block

def GetPaddleDebugNumAllowedOps():
    try:
        return int(os.getenv('PADDLE_DEBUG_NUM_ALLOWED_OPS'))
    except:
        return None

paddle_debug_num_allowed_ops = GetPaddleDebugNumAllowedOps()


if type(paddle_debug_num_allowed_ops) is not int:
    def EarlyReturn(block_idx, op_idx):
        return False      
else:
    def EarlyReturn(block_idx, op_idx):
        return op_idx >= paddle_debug_num_allowed_ops

class BlockEntries:
    def builtin_module_223_0_0(self, parameter_5, parameter_3, parameter_2, parameter_6, parameter_0, parameter_4, parameter_1, data_0, data_1):

        # pd_op.conv2d: (-1x2x-1x-1xf32) <- (-1x256x-1x-1xf32, 2x256x3x3xf32)
        conv2d_0 = paddle._C_ops.conv2d(data_0, parameter_0, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (4xi64) <- ()
        full_int_array_0 = [1, -1, 1, 1]

        # pd_op.reshape: (1x2x1x1xf32, 0x2xi64) <- (2xf32, 4xi64)
        reshape_0, reshape_1 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_1, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1x2x1x1xf32)
        add_0 = conv2d_0 + reshape_0

        # pd_op.multiply: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        multiply_0 = add_0 * parameter_2

        # pd_op.full: (1xf32) <- ()
        full_0 = paddle._C_ops.full([1], float('16'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.assign: (1xf32) <- (1xf32)
        assign_0 = full_0

        # pd_op.scale: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        scale_0 = paddle._C_ops.scale(multiply_0, full_0, float('0'), True)

        # pd_op.conv2d: (-1x2x-1x-1xf32) <- (-1x256x-1x-1xf32, 2x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(data_0, parameter_3, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x2x1x1xf32, 0x2xi64) <- (2xf32, 4xi64)
        reshape_2, reshape_3 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_4, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1x2x1x1xf32)
        add_1 = conv2d_1 + reshape_2

        # pd_op.multiply: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        multiply_1 = add_1 * parameter_2

        # pd_op.full: (1xf32) <- ()
        full_1 = paddle._C_ops.full([1], float('1'), paddle.float32, paddle.core.CPUPlace())

        # pd_op.scale: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        scale_1 = paddle._C_ops.scale(multiply_1, full_1, float('1'), True)

        # pd_op.elu: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32)
        elu_0 = paddle._C_ops.elu(scale_1, float('1'))

        # pd_op.scale: (-1x2x-1x-1xf32) <- (-1x2x-1x-1xf32, 1xf32)
        scale_2 = paddle._C_ops.scale(elu_0, assign_0, float('0'), True)

        # pd_op.conv2d: (-1x1x-1x-1xf32) <- (-1x256x-1x-1xf32, 1x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(data_0, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.reshape: (1x1x1x1xf32, 0x1xi64) <- (1xf32, 4xi64)
        reshape_4, reshape_5 = (lambda x, f: f(x))(paddle._C_ops.reshape(parameter_6, full_int_array_0), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.add: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, 1x1x1x1xf32)
        add_2 = conv2d_2 + reshape_4

        # pd_op.divide: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, -1xf32)
        divide_0 = add_2 / data_1

        # pd_op.sign: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32)
        sign_0 = paddle._C_ops.sign(divide_0)

        # pd_op.abs: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32)
        abs_0 = paddle._C_ops.abs(divide_0)

        # pd_op.floor: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32)
        floor_0 = paddle._C_ops.floor(abs_0)

        # pd_op.multiply: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, -1x1x-1x-1xf32)
        multiply_2 = sign_0 * floor_0

        # pd_op.multiply: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, -1xf32)
        multiply_3 = multiply_2 * data_1

        # pd_op.subtract: (-1x1x-1x-1xf32) <- (-1x1x-1x-1xf32, -1x1x-1x-1xf32)
        subtract_0 = add_2 - multiply_3

        # pd_op.full: (1xi32) <- ()
        full_2 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([-1x2x-1x-1xf32, -1x2x-1x-1xf32, -1x1x-1x-1xf32]) <- (-1x2x-1x-1xf32, -1x2x-1x-1xf32, -1x1x-1x-1xf32)
        combine_0 = [scale_0, scale_2, subtract_0]

        # pd_op.concat: (-1x5x-1x-1xf32) <- ([-1x2x-1x-1xf32, -1x2x-1x-1xf32, -1x1x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_2)

        # pd_op.flatten: (-1x5x-1xf32, 0x-1x5x-1x-1xf32) <- (-1x5x-1x-1xf32)
        flatten_0, flatten_1 = (lambda x, f: f(x))(paddle._C_ops.flatten(concat_0, 2, 3), lambda out: out if isinstance(out, (list, tuple)) else (out, None))

        # pd_op.transpose: (-1x-1x5xf32) <- (-1x5x-1xf32)
        transpose_0 = paddle._C_ops.transpose(flatten_0, [0, 2, 1])
        return conv2d_0, reshape_0, reshape_1, add_0, full_0, conv2d_1, reshape_2, reshape_3, add_1, full_1, scale_1, elu_0, assign_0, conv2d_2, reshape_4, reshape_5, add_2, divide_0, sign_0, floor_0, multiply_2, multiply_3, full_2, flatten_1, scale_0, scale_2, subtract_0, concat_0, transpose_0



def GetEnvVarEnableJit():
    enable_jit = os.getenv('PADDLE_DEBUG_ENABLE_JIT')
    return enable_jit not in {
        "0",
        "False",
        "false",
        "OFF",
    }

def GetEnvVarEnableCinn():
    enable_cinn = os.getenv('PADDLE_DEBUG_ENABLE_CINN')
    return enable_cinn not in {
        "0",
        "False",
        "false",
        "OFF",
    }


def GetTolerance(dtype):
    if dtype == np.float16:
        return GetFloat16Tolerance()
    if dtype == np.float32:
        return GetFloat32Tolerance()
    return 1e-6

def GetFloat16Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT16_TOL'))
    except:
        return 1e-3

def GetFloat32Tolerance():
    try:
        return float(os.getenv('PADDLE_DEBUG_FLOAT32_TOL'))
    except:
        return 1e-6

def IsInteger(dtype):
    return np.dtype(dtype).char in np.typecodes['AllInteger']


class CinnTestBase:
    def setUp(self):
        paddle.seed(2024)
        self.prepare_data()

    def _test_entry(self):
        dy_outs = self.entry(use_cinn=False)
        cinn_outs = self.entry(use_cinn=GetEnvVarEnableCinn())

        for cinn_out, dy_out in zip(cinn_outs, dy_outs):
          if type(cinn_out) is list and type(dy_out) is list:
            for x, y in zip(cinn_out, dy_out):
              self.assert_all_close(x, y)
          else:
            self.assert_all_close(cinn_out, dy_out)

    def assert_all_close(self, x, y):
        if (hasattr(x, "numpy") and hasattr(y, "numpy")):
            x_numpy = x.numpy()
            y_numpy = y.numpy()
            assert x_numpy.dtype == y_numpy.dtype
            if IsInteger(x_numpy.dtype):
                np.testing.assert_equal(x_numpy, y_numpy)
            else:
                tol = GetTolerance(x_numpy.dtype)
                np.testing.assert_allclose(x_numpy, y_numpy, atol=tol, rtol=tol)
        else:
            assert x == y

class ModuleOp(paddle.nn.Layer, BlockEntries):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_5, parameter_3, parameter_2, parameter_6, parameter_0, parameter_4, parameter_1, data_0, data_1):
        return self.builtin_module_223_0_0(parameter_5, parameter_3, parameter_2, parameter_6, parameter_0, parameter_4, parameter_1, data_0, data_1)

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_223_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_5
            paddle.uniform([1, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_2
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_6
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([2, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([2], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([2], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([1, 256, 64, 64], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.uniform([1], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_5
            paddle.static.InputSpec(shape=[1, 256, 3, 3], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[2, 256, 3, 3], dtype='float32'),
            # parameter_2
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_6
            paddle.static.InputSpec(shape=[1], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[2, 256, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[2], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[2], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[None, 256, None, None], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[None], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def entry(self, use_cinn):
        net = ModuleOp()
        if GetEnvVarEnableJit():
            net = self.apply_to_static(net, use_cinn)
        paddle.seed(2024)
        out = net(*self.inputs)
        return out

    def test_entry(self):
        if AthenaTryRunEnabled():
            if try_run_exit_code == 0:
                # All unittest cases passed.
                return
            if try_run_exit_code < 0:
                # program panicked.
                raise RuntimeError(f"panicked. panic stderr have been reported by the unittest `TestTryRun.test_panic`.")
        self._test_entry()

if __name__ == '__main__':
    unittest.main()