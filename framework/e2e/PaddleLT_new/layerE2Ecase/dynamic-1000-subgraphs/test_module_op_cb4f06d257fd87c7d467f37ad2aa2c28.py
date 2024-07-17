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
    return [20][block_idx] - 1 # number-of-ops-in-block

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

    def builtin_module_164_0_0(self, parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, data_3, data_0, data_1, data_2):

        # pd_op.bilinear_interp: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(data_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', False, 0)

        # pd_op.bilinear_interp: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, None, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(data_1, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

        # pd_op.bilinear_interp: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, None, None, None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(data_2, None, None, None, 'NCHW', -1, -1, -1, [float('8'), float('8')], 'bilinear', False, 0)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        # builtin.combine: ([-1x18x-1x-1xf32, -1x36x-1x-1xf32, -1x72x-1x-1xf32, -1x144x-1x-1xf32]) <- (-1x18x-1x-1xf32, -1x36x-1x-1xf32, -1x72x-1x-1xf32, -1x144x-1x-1xf32)
        combine_0 = [data_3, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2]

        # pd_op.concat: (-1x270x-1x-1xf32) <- ([-1x18x-1x-1xf32, -1x36x-1x-1xf32, -1x72x-1x-1xf32, -1x144x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x270x-1x-1xf32, 256x270x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(concat_0, parameter_0, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [2, 2]

        # pd_op.pool2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(conv2d_0, full_int_array_0, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [4, 4]

        # pd_op.pool2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(conv2d_0, full_int_array_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [8, 8]

        # pd_op.pool2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(conv2d_0, full_int_array_2, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [16, 16]

        # pd_op.pool2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(conv2d_0, full_int_array_3, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(conv2d_0, parameter_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_0, parameter_2, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(pool2d_1, parameter_3, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_2, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(pool2d_3, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')
        return bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3, pool2d_3, conv2d_1, conv2d_2, conv2d_3, conv2d_4, conv2d_5



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

class Block_builtin_module_164_0_0(paddle.nn.Layer, BlockEntries):
    def __init__(self):
        super().__init__()

    def forward(self, parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, data_3, data_0, data_1, data_2):
        args = [parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, data_3, data_0, data_1, data_2]
        for op_idx, op_func in enumerate(self.get_op_funcs()):
            if EarlyReturn(0, op_idx):
                return args
            args = op_func(*args)
        return args

    def get_op_funcs(self):
        return [
            self.op_bilinear_interp_0,
            self.op_bilinear_interp_1,
            self.op_bilinear_interp_2,
            self.op_full_0,
            self.op_combine_0,
            self.op_concat_0,
            self.op_conv2d_0,
            self.op_full_int_array_0,
            self.op_pool2d_0,
            self.op_full_int_array_1,
            self.op_pool2d_1,
            self.op_full_int_array_2,
            self.op_pool2d_2,
            self.op_full_int_array_3,
            self.op_pool2d_3,
            self.op_conv2d_1,
            self.op_conv2d_2,
            self.op_conv2d_3,
            self.op_conv2d_4,
            self.op_conv2d_5,
        ]

    def op_bilinear_interp_0(self, parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, data_3, data_0, data_1, data_2):
    
        # EarlyReturn(0, 0)

        # pd_op.bilinear_interp: (-1x36x-1x-1xf32) <- (-1x36x-1x-1xf32, None, None, None)
        bilinear_interp_0 = paddle._C_ops.bilinear_interp(data_0, None, None, None, 'NCHW', -1, -1, -1, [float('2'), float('2')], 'bilinear', False, 0)

        return [parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, data_3, data_1, data_2, bilinear_interp_0]

    def op_bilinear_interp_1(self, parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, data_3, data_1, data_2, bilinear_interp_0):
    
        # EarlyReturn(0, 1)

        # pd_op.bilinear_interp: (-1x72x-1x-1xf32) <- (-1x72x-1x-1xf32, None, None, None)
        bilinear_interp_1 = paddle._C_ops.bilinear_interp(data_1, None, None, None, 'NCHW', -1, -1, -1, [float('4'), float('4')], 'bilinear', False, 0)

        return [parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, data_3, data_2, bilinear_interp_0, bilinear_interp_1]

    def op_bilinear_interp_2(self, parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, data_3, data_2, bilinear_interp_0, bilinear_interp_1):
    
        # EarlyReturn(0, 2)

        # pd_op.bilinear_interp: (-1x144x-1x-1xf32) <- (-1x144x-1x-1xf32, None, None, None)
        bilinear_interp_2 = paddle._C_ops.bilinear_interp(data_2, None, None, None, 'NCHW', -1, -1, -1, [float('8'), float('8')], 'bilinear', False, 0)

        return [parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, data_3, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2]

    def op_full_0(self, parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, data_3, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2):
    
        # EarlyReturn(0, 3)

        # pd_op.full: (1xi32) <- ()
        full_0 = paddle._C_ops.full([1], float('1'), paddle.int32, paddle.core.CPUPlace())

        return [parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, data_3, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0]

    def op_combine_0(self, parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, data_3, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0):
    
        # EarlyReturn(0, 4)

        # builtin.combine: ([-1x18x-1x-1xf32, -1x36x-1x-1xf32, -1x72x-1x-1xf32, -1x144x-1x-1xf32]) <- (-1x18x-1x-1xf32, -1x36x-1x-1xf32, -1x72x-1x-1xf32, -1x144x-1x-1xf32)
        combine_0 = [data_3, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2]

        return [parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, combine_0]

    def op_concat_0(self, parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, combine_0):
    
        # EarlyReturn(0, 5)

        # pd_op.concat: (-1x270x-1x-1xf32) <- ([-1x18x-1x-1xf32, -1x36x-1x-1xf32, -1x72x-1x-1xf32, -1x144x-1x-1xf32], 1xi32)
        concat_0 = paddle._C_ops.concat(combine_0, full_0)

        return [parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0]

    def op_conv2d_0(self, parameter_2, parameter_3, parameter_0, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0):
    
        # EarlyReturn(0, 6)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x270x-1x-1xf32, 256x270x1x1xf32)
        conv2d_0 = paddle._C_ops.conv2d(concat_0, parameter_0, [1, 1], [0, 0], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0]

    def op_full_int_array_0(self, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0):
    
        # EarlyReturn(0, 7)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_0 = [2, 2]

        return [parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0]

    def op_pool2d_0(self, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0):
    
        # EarlyReturn(0, 8)

        # pd_op.pool2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_0 = paddle._C_ops.pool2d(conv2d_0, full_int_array_0, [2, 2], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        return [parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0]

    def op_full_int_array_1(self, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0):
    
        # EarlyReturn(0, 9)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_1 = [4, 4]

        return [parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1]

    def op_pool2d_1(self, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1):
    
        # EarlyReturn(0, 10)

        # pd_op.pool2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_1 = paddle._C_ops.pool2d(conv2d_0, full_int_array_1, [4, 4], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        return [parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1]

    def op_full_int_array_2(self, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1):
    
        # EarlyReturn(0, 11)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_2 = [8, 8]

        return [parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2]

    def op_pool2d_2(self, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2):
    
        # EarlyReturn(0, 12)

        # pd_op.pool2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_2 = paddle._C_ops.pool2d(conv2d_0, full_int_array_2, [8, 8], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        return [parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2]

    def op_full_int_array_3(self, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2):
    
        # EarlyReturn(0, 13)

        # pd_op.full_int_array: (2xi64) <- ()
        full_int_array_3 = [16, 16]

        return [parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3]

    def op_pool2d_3(self, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3):
    
        # EarlyReturn(0, 14)

        # pd_op.pool2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 2xi64)
        pool2d_3 = paddle._C_ops.pool2d(conv2d_0, full_int_array_3, [16, 16], [0, 0], False, True, 'NCHW', 'avg', False, False, 'EXPLICIT')

        return [parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3, pool2d_3]

    def op_conv2d_1(self, parameter_2, parameter_3, parameter_5, parameter_4, parameter_1, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3, pool2d_3):
    
        # EarlyReturn(0, 15)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_1 = paddle._C_ops.conv2d(conv2d_0, parameter_1, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_2, parameter_3, parameter_5, parameter_4, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3, pool2d_3, conv2d_1]

    def op_conv2d_2(self, parameter_2, parameter_3, parameter_5, parameter_4, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3, pool2d_3, conv2d_1):
    
        # EarlyReturn(0, 16)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_2 = paddle._C_ops.conv2d(pool2d_0, parameter_2, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_3, parameter_5, parameter_4, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3, pool2d_3, conv2d_1, conv2d_2]

    def op_conv2d_3(self, parameter_3, parameter_5, parameter_4, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3, pool2d_3, conv2d_1, conv2d_2):
    
        # EarlyReturn(0, 17)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_3 = paddle._C_ops.conv2d(pool2d_1, parameter_3, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_5, parameter_4, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3, pool2d_3, conv2d_1, conv2d_2, conv2d_3]

    def op_conv2d_4(self, parameter_5, parameter_4, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3, pool2d_3, conv2d_1, conv2d_2, conv2d_3):
    
        # EarlyReturn(0, 18)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_4 = paddle._C_ops.conv2d(pool2d_2, parameter_4, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [parameter_5, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3, pool2d_3, conv2d_1, conv2d_2, conv2d_3, conv2d_4]

    def op_conv2d_5(self, parameter_5, bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3, pool2d_3, conv2d_1, conv2d_2, conv2d_3, conv2d_4):
    
        # EarlyReturn(0, 19)

        # pd_op.conv2d: (-1x256x-1x-1xf32) <- (-1x256x-1x-1xf32, 256x256x3x3xf32)
        conv2d_5 = paddle._C_ops.conv2d(pool2d_3, parameter_5, [1, 1], [1, 1], 'EXPLICIT', [1, 1], 1, 'NCHW')

        return [bilinear_interp_0, bilinear_interp_1, bilinear_interp_2, full_0, concat_0, conv2d_0, full_int_array_0, pool2d_0, full_int_array_1, pool2d_1, full_int_array_2, pool2d_2, full_int_array_3, pool2d_3, conv2d_1, conv2d_2, conv2d_3, conv2d_4, conv2d_5]

@unittest.skipIf(need_skip, skip_message)
class Test_builtin_module_164_0_0(CinnTestBase, unittest.TestCase):
    def prepare_data(self):
        self.inputs = [
            # parameter_2
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_3
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_0
            paddle.uniform([256, 270, 1, 1], dtype='float32', min=0, max=0.5),
            # parameter_5
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_4
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # parameter_1
            paddle.uniform([256, 256, 3, 3], dtype='float32', min=0, max=0.5),
            # data_3
            paddle.uniform([1, 18, 176, 264], dtype='float32', min=0, max=0.5),
            # data_0
            paddle.uniform([1, 36, 88, 132], dtype='float32', min=0, max=0.5),
            # data_1
            paddle.uniform([1, 72, 44, 66], dtype='float32', min=0, max=0.5),
            # data_2
            paddle.uniform([1, 144, 22, 33], dtype='float32', min=0, max=0.5),
        ]
        for input in self.inputs:
            input.stop_gradient = True

    def apply_to_static(self, net, use_cinn):
        build_strategy = paddle.static.BuildStrategy()
        input_spec = [
            # parameter_2
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_3
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_0
            paddle.static.InputSpec(shape=[256, 270, 1, 1], dtype='float32'),
            # parameter_5
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_4
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # parameter_1
            paddle.static.InputSpec(shape=[256, 256, 3, 3], dtype='float32'),
            # data_3
            paddle.static.InputSpec(shape=[None, 18, None, None], dtype='float32'),
            # data_0
            paddle.static.InputSpec(shape=[None, 36, None, None], dtype='float32'),
            # data_1
            paddle.static.InputSpec(shape=[None, 72, None, None], dtype='float32'),
            # data_2
            paddle.static.InputSpec(shape=[None, 144, None, None], dtype='float32'),
        ]
        build_strategy.build_cinn_pass = use_cinn
        return paddle.jit.to_static(
            net,
            input_spec=input_spec,
            build_strategy=build_strategy,
            full_graph=True,
        )

    def entry(self, use_cinn):
        net = Block_builtin_module_164_0_0()
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