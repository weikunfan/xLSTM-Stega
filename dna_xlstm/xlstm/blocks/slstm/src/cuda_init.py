# Copyright 2023 JKU Linz
# Korbinian Poeppel

import os
from typing import Sequence, Union
import logging

import time
import random

import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import load as _load
from torch.utils import cpp_extension

# import torch.nn.functional as F
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)

# print("INCLUDE:", torch.utils.cpp_extension.include_paths(cuda=True))
# print("C++ compat", torch.utils.cpp_extension.check_compiler_abi_compatibility("g++"))
# print("C compat", torch.utils.cpp_extension.check_compiler_abi_compatibility("gcc"))

LOGGER = logging.getLogger(__name__)


def defines_to_cflags(defines=Union[dict[str, Union[int, str]], Sequence[tuple[str, Union[str, int]]]]):
    cflags = []
    print(defines)
    if isinstance(defines, dict):
        defines = defines.items()
    for key, val in defines:
        cflags.append(f"-D{key}={str(val)}")
    return cflags


curdir = os.path.dirname(__file__)
# print(curdir)

#CUDA_INCLUDE = os.environ.get("CUDA_INCLUDE", "/usr/lib")
os.environ["CUDA_LIB"] = os.path.join(os.path.split(cpp_extension.include_paths(cuda=True)[-1])[0], "lib") #TODO JS: Change
#print(os.environ.get("LD_LIBRARY_PATH", ""))
#print(os.environ["CUDA_LIB"])


def load(*, name, sources, extra_cflags=(), extra_cuda_cflags=(), **kwargs):
    suffix = ""
    for flag in extra_cflags:
        pref = [st[0] for st in flag[2:].split("=")[0].split("_")]
        if len(pref) > 1:
            pref = pref[1:]
        suffix += "".join(pref)
        value = flag[2:].split("=")[1].replace("-", "m").replace(".", "d")
        value_map = {"float": "f", "__half": "h", "__nv_bfloat16": "b", "true": "1", "false": "0"}
        if value in value_map:
            value = value_map[value]
        suffix += value
    if suffix:
        suffix = "_" + suffix
    suffix = suffix[:64]

    extra_cflags = list(extra_cflags) + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    ]

    myargs = {
        "verbose": True,
        "with_cuda": True,
        "extra_ldflags": [f"-L{os.environ['CUDA_LIB']}", "-lcublas"],
        "extra_cflags": [*extra_cflags],
        "extra_cuda_cflags": [
            # "-gencode",
            # "arch=compute_70,code=compute_70",
            # "-dbg=1",
            '-Xptxas="-v"',
            "-gencode",
            "arch=compute_80,code=compute_80",
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
            *extra_cflags,
            *extra_cuda_cflags,
        ],
    }
    print(myargs)
    myargs.update(**kwargs)
    # add random waiting time to minimize deadlocks because of badly managed multicompile of pytorch ext
    time.sleep(random.random() * 10)
    LOGGER.info(f"Before compilation and loading of {name}.")
    mod = _load(name + suffix, sources, **myargs)
    LOGGER.info(f"After compilation and loading of {name}.")
    return mod

