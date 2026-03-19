from typing import TypeAlias
import numpy as np
import numpy.typing as npt
import ml_dtypes
import thor

dtype: TypeAlias = npt.DTypeLike

fp16 = np.float16
fp32 = np.float32
bf16 = ml_dtypes.bfloat16
fp8_e4m3 = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2


def from_thor(dtype: thor.DataType):
    if dtype == thor.DataType.fp16:
        return fp16
    if dtype == thor.DataType.bf16:
        return bf16
    if dtype == thor.DataType.fp8_e4m3:
        return fp8_e4m3
    if dtype == thor.DataType.fp8_e5m2:
        return fp8_e5m2
    if dtype == thor.DataType.fp32:
        return fp32
    raise TypeError(f"Unsupported Thor dtype: {dtype}")


def to_thor(dtype):
    if dtype == np.float16:
        return thor.DataType.fp16
    if dtype == np.float32:
        return thor.DataType.fp32
    if dtype == ml_dtypes.bfloat16:
        return thor.DataType.bf16
    if dtype == ml_dtypes.float8_e4m3fn:
        return thor.DataType.fp8_e4m3
    if dtype == ml_dtypes.float8_e5m2:
        return thor.DataType.fp8_e5m2
    raise TypeError(f"Unsupported NumPy/ml_dtypes dtype: {dtype}")
