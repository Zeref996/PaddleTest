#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
/***************************************************************************
  *
  * Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
  * @file test_dist_save_state_dict.py
  * @author liujie44@baidu.com
  * @date 2024-03-04
  * @brief
  *
  **************************************************************************/
"""
import os
import subprocess
from utils import run_priority

os.system("export CUDA_VISIBLE_DEVICES=0,1")


class TestApi(object):
    """TestApi"""

    def test_save_state_dict(self):
        """test_save_state_dict"""
        cmd = "python -m paddle.distributed.launch --devices 0,1 --job_id save_state_dict dist_save_state_dict.py"
        pro = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = pro.communicate()
        print(out)
        pro.wait()
        pro.returncode == 0
        assert str(out).find("Error") == -1
        assert str(err).find("Error") == -1