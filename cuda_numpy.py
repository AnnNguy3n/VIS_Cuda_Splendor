from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float64


@cuda.jit(device=True)
def shuffle(_dst, _len, _rng_states, _idx):
    for i in range(_len):
        swap_idx = int(xoroshiro128p_uniform_float64(_rng_states, _idx) * _len)
        temp = _dst[i]
        _dst[i] = _dst[swap_idx]
        _dst[swap_idx] = temp


@cuda.jit(device=True)
def assign(_dst, _src, _len):
    for i in range(_len): _dst[i] = _src[i]


@cuda.jit(device=True)
def any(_arr, _len, _value):
    for i in range(_len):
        if _arr[i] == _value: return True

    return False


@cuda.jit(device=True)
def all(_arr, _len, _value):
    for i in range(_len):
        if _arr[i] != _value: return False

    return True


@cuda.jit(device=True)
def all_greater(_arr, _len, _value):
    for i in range(_len):
        if _arr[i] <= _value: return False

    return True


@cuda.jit(device=True)
def all_lower(_arr, _len, _value):
    for i in range(_len):
        if _arr[i] >= _value: return False

    return True


@cuda.jit(device=True)
def sum(_arr, _len):
    sum_ = 0.0
    for i in range(_len): sum_ += _arr[i]

    return sum_


@cuda.jit(device=True)
def index(_arr, _len, _value):
    for i in range(_len):
        if _arr[i] == _value: return i

    return -9223372036854775808


@cuda.jit(device=True)
def sum_product(_a, _b, _len):
    sum_ = 0.0
    for i in range(_len): sum_ += _a[i] * _b[i]

    return sum_


@cuda.jit(device=True)
def vector_add(_a, _b, _res, _len):
    for i in range(_len): _res[i] = _a[i] + _b[i]
