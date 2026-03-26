from typing import TypeAlias
import numpy as np
import numpy.typing as npt
import ml_dtypes
import thor

dtype: TypeAlias = npt.DTypeLike

bool_ = np.bool_

int8 = np.int8
uint8 = np.uint8
int16 = np.int16
uint16 = np.uint16
int32 = np.int32
uint32 = np.uint32
int64 = np.int64
uint64 = np.uint64

fp16 = np.float16
fp32 = np.float32
fp64 = np.float64
bf16 = ml_dtypes.bfloat16
fp8_e4m3 = ml_dtypes.float8_e4m3fn
fp8_e5m2 = ml_dtypes.float8_e5m2


def from_thor(dtype: thor.DataType):
    if dtype == thor.DataType.bool:
        return bool_
    if dtype == thor.DataType.packed_bool:
        return uint8

    if dtype == thor.DataType.int8:
        return int8
    if dtype == thor.DataType.uint8:
        return uint8
    if dtype == thor.DataType.int16:
        return int16
    if dtype == thor.DataType.uint16:
        return uint16
    if dtype == thor.DataType.int32:
        return int32
    if dtype == thor.DataType.uint32:
        return uint32
    if dtype == thor.DataType.int64:
        return int64
    if dtype == thor.DataType.uint64:
        return uint64

    if dtype == thor.DataType.fp16:
        return fp16
    if dtype == thor.DataType.fp32:
        return fp32
    if dtype == thor.DataType.fp64:
        return fp64
    if dtype == thor.DataType.bf16:
        return bf16
    if dtype == thor.DataType.fp8_e4m3:
        return fp8_e4m3
    if dtype == thor.DataType.fp8_e5m2:
        return fp8_e5m2
    raise TypeError(f"Unsupported Thor dtype: {dtype}")


def to_thor(dtype):
    if dtype == np.bool_:
        return thor.DataType.bool

    if dtype == np.int8:
        return thor.DataType.int8
    if dtype == np.uint8:
        return thor.DataType.uint8
    if dtype == np.int16:
        return thor.DataType.int16
    if dtype == np.uint16:
        return thor.DataType.uint16
    if dtype == np.int32:
        return thor.DataType.int32
    if dtype == np.uint32:
        return thor.DataType.uint32
    if dtype == np.int64:
        return thor.DataType.int64
    if dtype == np.uint64:
        return thor.DataType.uint64

    if dtype == np.float16:
        return thor.DataType.fp16
    if dtype == np.float32:
        return thor.DataType.fp32
    if dtype == np.float64:
        return thor.DataType.fp64
    if dtype == ml_dtypes.bfloat16:
        return thor.DataType.bf16
    if dtype == ml_dtypes.float8_e4m3fn:
        return thor.DataType.fp8_e4m3
    if dtype == ml_dtypes.float8_e5m2:
        return thor.DataType.fp8_e5m2
    if dtype == np.uint32:
        return thor.DataType.uint32
    raise TypeError(f"Unsupported NumPy/ml_dtypes dtype: {dtype}")
