from cuda_env import *
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states
import time


@cuda.jit(device=True)
def get_action_max(weights, actions):
    max_ = -9223372036854775808
    actions_max = -9223372036854775808
    for i in range(actions[41]):
        action = actions[i]
        if weights[action] > max_:
            max_ = weights[action]
            actions_max = action

    return actions_max


@cuda.jit(device=True)
def check_win(env, p_idx):
    temp = 15*p_idx
    if env[34+temp] < 15:
        return 0

    check = False
    for i in range(4):
        if i != p_idx:
            temp_ = 15*i
            if env[34+temp_] >= env[34+temp]:
                check = True
                if env[34+temp_] > env[34+temp]:
                    return 0

    if not check:
        return 1

    sum_p_per_stocks = cuPy.sum(env[29+temp:34+temp], 5)
    for i in range(4):
        if i != p_idx:
            temp_ = 15*i
            sum_ = cuPy.sum(env[29+temp_:34+temp_], 5)
            if env[34+temp_] == env[34+temp] and sum_ < sum_p_per_stocks:
                return 0

    return 1


@cuda.jit()
def __Train__(n_threads, env_infors, rng_states, arr_actions, NORMAL_CARD, NOBLE_CARD, per_0, per_1s, size, result):
    tx = cuda.grid(1)
    if tx < n_threads:
        result[tx] = 0
        env_infor = env_infors[tx]
        env = env_infor[0:90]
        lv1 = env_infor[90:131]
        lv2 = env_infor[131:162]
        lv3 = env_infor[162:183]
        noble = env_infor[183:193]
        state = env_infor[193:460]

        actions = arr_actions[tx]

        per_1 = per_1s[tx]

        init_env(env, lv1, lv2, lv3, noble, rng_states, tx)
        p_idx = int(xoroshiro128p_uniform_float64(rng_states, tx) * 4)
        end = False
        while not end:
            cur_idx = env[83] % 4
            if cur_idx != p_idx:
                while env[83] % 4 == cur_idx:
                    get_valid_actions(env, actions, lv1, lv2, lv3, NORMAL_CARD)
                    if actions[41] == 0:
                        print("Tập action của bot hệ thống bị rỗng")
                        raise

                    if actions[41] == 1:
                        choose = 0
                    else:
                        choose = int(xoroshiro128p_uniform_float64(rng_states, tx) * actions[41])

                    action = actions[choose]
                    step_env(action, env, lv1, lv2, lv3, NORMAL_CARD, NOBLE_CARD, rng_states, tx)
            else:
                while env[83] % 4 == cur_idx:
                    get_valid_actions(env, actions, lv1, lv2, lv3, NORMAL_CARD)
                    if actions[41] == 0:
                        print("Tập action của bot hệ thống bị rỗng")
                        raise

                    get_agent_state(env, lv1, lv2, lv3, state, NORMAL_CARD, NOBLE_CARD)

                    index = 0
                    for i in range(size):
                        att = per_0[i]
                        if cuPy.sum_product(state, att, 267) > 1.0:
                            index += 2**i

                    weights = per_1[41*index:41*(index+1)]
                    action = get_action_max(weights, actions)
                    step_env(action, env, lv1, lv2, lv3, NORMAL_CARD, NOBLE_CARD, rng_states, tx)

            if env[83] == 400:
                end = True
                break

            if env[83] % 4 == 0:
                for i in range(34, 80, 15):
                    if env[i] >= 15:
                        end = True
                        break

        win = check_win(env, p_idx)
        result[tx] = win + 1


@cuda.jit(debug=True, opt=False)
def __Update__(per_1s, per_2s, is_end, size, n_threads, rng_states):
    tx = cuda.grid(1)
    if tx < n_threads:
        n = 2**size
        r_idx = tx // n
        c_idx = tx % n
        per_1 = per_1s[r_idx][41*c_idx:41*(c_idx+1)]
        per_2 = per_2s[r_idx][41*c_idx:41*(c_idx+1)]

        if is_end[r_idx] == 1: # Thua
            cuPy.shuffle(per_1, 41, rng_states, r_idx)
        else: # Thắng
            cuPy.vector_add(per_2, per_1, per_2, 41)


def Train(n_threads, n_cycles, arr_attributes):
    '''
    n_threads: số luồng, tương ứng số trận chạy song song
    n_cycles: số trận mà mỗi luồng xử lí
    arr_attributes: mảng các tính chất, mỗi tính chất là một array có độ dài bằng 267 (bằng độ dài của state).
    '''

    size = arr_attributes.shape[0]

    if size > 10:
        print("Hơn 10 thuộc tính có thể dẫn đến tràn VRAM.")
    
    per_0 = cuda.to_device(arr_attributes)

    temp = np.arange(41, dtype=np.float64) + 1.0
    temp_1 = np.array([temp] * (2**size)).flatten()
    per_1 = cuda.to_device(np.array([temp_1] * n_threads))
    del temp
    del temp_1

    per_2 = cuda.to_device(np.ones((n_threads, 41*(2**size)), np.float64))

    result = cuda.to_device(np.full(n_threads, 1, dtype=np.int8))

    n_threads_per_block = 32
    n_update_threads = n_threads * (2**size)
    n_update_blocks = n_update_threads // n_threads_per_block + 1
    if n_update_threads % n_threads_per_block == 0.0:
        n_update_blocks -= 1
    
    rng_states = create_xoroshiro128p_states(n_update_threads, time.time()*1e6)
    __Update__[n_update_blocks, n_threads_per_block](per_1, per_2, result, size, n_update_threads, rng_states)
    cuda.synchronize()

    n_blocks = n_threads // n_threads_per_block + 1
    if n_threads % n_threads_per_block == 0.0:
        n_blocks -= 1
    
    env_infor = np.full(460, 0, np.int16)
    env_infor[90:131] = np.arange(41)
    env_infor[131:162] = np.arange(40, 71)
    env_infor[162:183] = np.arange(70, 91)
    env_infor[183:193] = np.arange(10)
    d_env_infor = cuda.to_device(np.array([env_infor]*n_threads))
    del env_infor

    arr_actions = cuda.device_array((n_threads, 42), np.int16)

    NORMAL = cuda.to_device(np.array([[0, 2, 2, 2, 0, 0, 0], [0, 2, 3, 0, 0, 0, 0], [0, 2, 1, 1, 0, 2, 1], [0, 2, 0, 1, 0, 0, 2], [0, 2, 0, 3, 1, 0, 1], [0, 2, 1, 1, 0, 1, 1], [1, 2, 0, 0, 0, 4, 0], [0, 2, 2, 1, 0, 2, 0], [0, 1, 2, 0, 2, 0, 1], [0, 1, 0, 0, 2, 2, 0], [0, 1, 1, 0, 1, 1, 1], [0, 1, 2, 0, 1, 1, 1], [0, 1, 1, 1, 3, 0, 0], [0, 1, 0, 0, 0, 2, 1], [0, 1, 0, 0, 0, 3, 0], [1, 1, 4, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 4], [0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 1, 1, 1, 2], [0, 0, 0, 0, 1, 2, 2], [0, 0, 1, 0, 0, 3, 1], [0, 0, 2, 0, 0, 0, 2], [0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 2, 1, 0, 0], [0, 4, 0, 2, 2, 1, 0], [0, 4, 1, 1, 2, 1, 0], [0, 4, 0, 1, 0, 1, 3], [1, 4, 0, 0, 4, 0, 0], [0, 4, 0, 2, 0, 2, 0], [0, 4, 2, 0, 0, 1, 0], [0, 4, 1, 1, 1, 1, 0], [0, 4, 0, 3, 0, 0, 0], [0, 3, 1, 0, 2, 0, 0], [0, 3, 1, 1, 1, 0, 1], [1, 3, 0, 4, 0, 0, 0], [0, 3, 1, 2, 0, 0, 2], [0, 3, 0, 0, 3, 0, 0], [0, 3, 0, 0, 2, 0, 2], [0, 3, 3, 0, 1, 1, 0], [0, 3, 1, 2, 1, 0, 1], [1, 2, 0, 3, 0, 2, 2], [2, 2, 0, 2, 0, 1, 4], [1, 2, 3, 0, 2, 0, 3], [2, 2, 0, 5, 3, 0, 0], [2, 2, 0, 0, 5, 0, 0], [3, 2, 0, 0, 6, 0, 0], [3, 1, 0, 6, 0, 0, 0], [2, 1, 1, 0, 0, 4, 2], [2, 1, 0, 5, 0, 0, 0], [2, 1, 0, 3, 0, 0, 5], [1, 1, 0, 2, 3, 3, 0], [1, 1, 3, 2, 2, 0, 0], [3, 0, 6, 0, 0, 0, 0], [2, 0, 0, 0, 0, 5, 3], [2, 0, 0, 0, 0, 5, 0], [2, 0, 0, 4, 2, 0, 1], [1, 0, 2, 3, 0, 3, 0], [1, 0, 2, 0, 0, 3, 2], [3, 4, 0, 0, 0, 0, 6], [2, 4, 5, 0, 0, 3, 0], [2, 4, 5, 0, 0, 0, 0], [1, 4, 3, 3, 0, 0, 2], [1, 4, 2, 0, 3, 2, 0], [2, 4, 4, 0, 1, 2, 0], [1, 3, 0, 2, 2, 0, 3], [1, 3, 0, 0, 3, 2, 3], [2, 3, 2, 1, 4, 0, 0], [2, 3, 3, 0, 5, 0, 0], [2, 3, 0, 0, 0, 0, 5], [3, 3, 0, 0, 0, 6, 0], [4, 2, 0, 7, 0, 0, 0], [4, 2, 0, 6, 3, 0, 3], [5, 2, 0, 7, 3, 0, 0], [3, 2, 3, 3, 0, 3, 5], [3, 1, 3, 0, 3, 5, 3], [4, 1, 0, 0, 0, 0, 7], [5, 1, 0, 3, 0, 0, 7], [4, 1, 0, 3, 0, 3, 6], [3, 0, 0, 5, 3, 3, 3], [4, 0, 0, 0, 7, 0, 0], [5, 0, 3, 0, 7, 0, 0], [4, 0, 3, 3, 6, 0, 0], [5, 4, 0, 0, 0, 7, 3], [3, 4, 5, 3, 3, 3, 0], [4, 4, 0, 0, 0, 7, 0], [4, 4, 3, 0, 0, 6, 3], [3, 3, 3, 3, 5, 0, 3], [5, 3, 7, 0, 0, 3, 0], [4, 3, 6, 0, 3, 3, 0], [4, 3, 7, 0, 0, 0, 0]], dtype=np.int16))
    NOBLE = cuda.to_device(np.array([[3, 0, 4, 4, 0, 0], [3, 3, 0, 3, 3, 0], [3, 3, 3, 3, 0, 0], [3, 3, 0, 0, 3, 3], [3, 0, 3, 0, 3, 3], [3, 4, 0, 4, 0, 0], [3, 4, 0, 0, 4, 0], [3, 0, 3, 3, 0, 3], [3, 0, 4, 0, 0, 4], [3, 0, 0, 0, 4, 4]], dtype=np.int16))

    for _n in range(n_cycles):

        rng_states = create_xoroshiro128p_states(n_update_threads, time.time()*1e6)
        __Train__[n_blocks, n_threads_per_block](n_threads, d_env_infor, rng_states, arr_actions, NORMAL, NOBLE, per_0, per_1, size, result)
        cuda.synchronize()
        __Update__[n_update_blocks, n_threads_per_block](per_1, per_2, result, size, n_update_threads, rng_states)
        cuda.synchronize()
    
    return np.sum(per_2.copy_to_host(), axis=0).reshape((2**size, 41))
