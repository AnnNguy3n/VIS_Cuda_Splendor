from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float64
import cuda_numpy as cuPy


@cuda.jit(device=True)
def init_env(env, lv1, lv2, lv3, noble, rng_states, match_idx):
    cuPy.shuffle(noble, 10, rng_states, match_idx)
    cuPy.shuffle(lv1[0:40], 40, rng_states, match_idx)
    cuPy.shuffle(lv2[0:30], 30, rng_states, match_idx)
    cuPy.shuffle(lv3[0:20], 20, rng_states, match_idx)

    lv1[40] = 4
    lv2[30] = 4
    lv3[20] = 4

    env[0:5] = 7
    env[5] = 5
    cuPy.assign(env[6:11], noble[0:5], 5)
    cuPy.assign(env[11:15], lv1[0:4], 4)
    cuPy.assign(env[15:19], lv2[0:4], 4)
    cuPy.assign(env[19:23], lv3[0:4], 4)

    for pIdx in range(4):
        temp_ = 15*pIdx
        env[23+temp_:35+temp_] = 0
        env[35+temp_:38+temp_] = -1

    env[83:90] = 0


@cuda.jit(device=True)
def check_buy_card(gems, perGems, price):
    sum_ = 0
    for i in range(5):
        if price[i] > (gems[i] + perGems[i]): sum_ += (price[i] - (gems[i] + perGems[i]))

    if sum_ <= gems[5]: return True
    return False


@cuda.jit(device=True)
def get_valid_actions(env, actions, lv1, lv2, lv3, NORMAL_CARD):
    n = 0
    boardStocks = env[0:6]
    takenStocks = env[84:89]
    if cuPy.any(takenStocks, 5, 1):
        s_ = cuPy.sum(takenStocks, 5)
        if s_ == 1:
            t_ = cuPy.index(takenStocks, 5, 1)
            if boardStocks[t_] >= 3: t_ = -1

            t1_ = -1
        else:
            t_ = cuPy.index(takenStocks, 5, 1)
            t1_ = cuPy.index(takenStocks[t_+1:5], 5-t_-1, 1) + t_ + 1

        for i in range(5):
            if i != t_ and i != t1_ and boardStocks[i] > 0:
                actions[n] = i
                n += 1

        actions[41] = n
        return

    p_idx = env[83] % 4
    temp = 15*p_idx
    p_stocks = env[23+temp:29+temp]

    if cuPy.sum(p_stocks, 6) > 10:
        for i in range(5):
            if p_stocks[i] > 0:
                actions[n] = 35 + i
                n += 1

        actions[41] = n
        return

    for i in range(5):
        if boardStocks[i] > 0:
            actions[n] = i
            n += 1

    p_reserve = env[35+temp:38+temp]
    check_reserve = cuPy.any(p_reserve, 3, -1)
    p_per_stocks = env[29+temp:34+temp]
    for i in range(12):
        card_id = env[11+i]
        if card_id != -1:
            if check_reserve:
                actions[n] = 20 + i
                n += 1

            card_price = NORMAL_CARD[card_id][2:7]
            if check_buy_card(p_stocks, p_per_stocks, card_price):
                actions[n] = 5 + i
                n += 1

    for i in range(3):
        card_id = p_reserve[i]
        if card_id != -1:
            card_price = NORMAL_CARD[card_id][2:7]
            if check_buy_card(p_stocks, p_per_stocks, card_price):
                actions[n] = 17 + i
                n += 1

    if check_reserve:
        if lv1[40] < 40:
            actions[n] = 32
            n += 1
        if lv2[30] < 30:
            actions[n] = 33
            n += 1
        if lv3[20] < 20:
            actions[n] = 34
            n += 1

    if n == 0:
        actions[0] = 40
        n = 1

    actions[41] = n
    return


@cuda.jit(device=True)
def check_noble(per_gem, price):
    for i in range(5):
        if price[i] > per_gem[i]: return False

    return True


@cuda.jit(device=True)
def open_card(env, lv1, lv2, lv3, cardId, posE):
    if cardId < 40:
        if lv1[40] < 40:
            env[posE] = lv1[lv1[40]]
            lv1[40] += 1
        else: env[posE] = -1
    elif cardId < 70:
        if lv2[30] < 30:
            env[posE] = lv2[lv2[30]]
            lv2[30] += 1
        else: env[posE] = -1
    else:
        if lv3[20] < 20:
            env[posE] = lv3[lv3[20]]
            lv3[20] += 1
        else: env[posE] = -1


@cuda.jit(device=True)
def step_env(action, env, lv1, lv2, lv3, NORMAL_CARD, NOBLE_CARD, rng_states, match_idx):
    pIdx = env[83] % 4
    temp_ = 15*pIdx
    pStocks = env[23+temp_:29+temp_]
    bStocks = env[0:6]
    pPerStocks = env[29+temp_:34+temp_]
    takenStocks = env[84:89]

    if action < 5:
        takenStocks[action] += 1
        pStocks[action] += 1
        bStocks[action] -= 1

        check_ = False
        s_ = cuPy.sum(takenStocks, 5)
        if s_ == 1:
            if bStocks[action] < 3 and (cuPy.sum(bStocks[0:5], 5) - bStocks[action]) == 0: check_ = True
        elif s_ == 2:
            if cuPy.index(takenStocks, 5, 2) != 9223372036854775807: check_ = True

            if not check_:
                t_ = cuPy.index(takenStocks, 5, 1)
                t1_ = cuPy.index(takenStocks[t_+1:5], 5-t_-1, 1) + t_ + 1

                if cuPy.sum(bStocks[0:5], 5) - bStocks[t_] - bStocks[t1_] == 0: check_ = True
        else: check_ = True

        if check_:
            takenStocks[:] = 0
            if cuPy.sum(pStocks, 6) <= 10: env[83] += 1

    elif action >= 35 and action < 40:
        gem = action - 35
        pStocks[gem] -= 1
        bStocks[gem] += 1

        if cuPy.sum(pStocks, 6) <= 10: env[83] += 1

    elif action >= 20 and action < 35:
        p_reserve = env[35+temp_:38+temp_]
        posP = cuPy.index(p_reserve, 3, -1) + 35 + temp_

        if bStocks[5] > 0:
            pStocks[5] += 1
            bStocks[5] -= 1

        if action == 32:
            env[posP] = lv1[lv1[40]]
            lv1[40] += 1
        elif action == 33:
            env[posP] = lv2[lv2[30]]
            lv2[30] += 1
        elif action == 34:
            env[posP] = lv3[lv3[20]]
            lv3[20] += 1
        else:
            posE = action - 9
            cardId = env[posE]
            env[posP] = cardId

            open_card(env, lv1, lv2, lv3, cardId, posE)

        if cuPy.sum(pStocks, 6) <= 10: env[83] += 1

    elif action >= 5 and action < 20:
        if action < 17: posE = action + 6
        else: posE = 18 + temp_ + action

        cardId = env[posE]
        cardIn4 = NORMAL_CARD[cardId]
        price = cardIn4[2:7]

        for i in range(5):
            if price[i] > pPerStocks[i]:
                nl_mat = price[i] - pPerStocks[i]
                if nl_mat <= pStocks[i]: nl_g = 0
                else:
                    nl_g = nl_mat - pStocks[i]
                    nl_mat = pStocks[i]

                pStocks[i] -= nl_mat
                pStocks[5] -= nl_g
                bStocks[i] += nl_mat
                bStocks[5] += nl_g

        if action < 17: open_card(env, lv1, lv2, lv3, cardId, posE)
        else: env[posE] = -1

        env[34+temp_] += cardIn4[0]
        pPerStocks[cardIn4[1]] += 1
        env[83] += 1

    else: env[83] += 1

    if cuPy.all(takenStocks, 5, 0) and cuPy.all_greater(pPerStocks, 5, 2):
        n = 0
        for i in range(5):
            nobleId = env[6+i]
            if nobleId != -1:
                nobleIn4 = NOBLE_CARD[nobleId]
                price = nobleIn4[1:6]
                if check_noble(pPerStocks, price):
                    takenStocks[n] = i
                    n += 1

        if n > 0:
            if n == 1: n_choice = 0
            else: n_choice = int(xoroshiro128p_uniform_float64(rng_states, match_idx)*n)

            noble_pos = takenStocks[n_choice]
            env[6+noble_pos] = -1
            env[34+temp_] += 3
            takenStocks[:] = 0


@cuda.jit(device=True)
def get_agent_state(env, lv1, lv2, lv3, state, NORMAL_CARD, NOBLE_CARD):
    state[:] = 0.0

    cuPy.assign(state[0:6], env[0:6], 6)

    for i in range(5):
        nobleId = env[6+i]
        if nobleId != -1:
            temp_ = 6*i
            cuPy.assign(state[6+temp_:12+temp_], NOBLE_CARD[nobleId], 6)

    for i in range(12):
        cardId = env[11+i]
        if cardId != -1:
            cardIn4 = NORMAL_CARD[cardId]
            temp_ = 11*i
            state[36+temp_] = cardIn4[0]
            state[37+temp_+cardIn4[1]] = 1
            cuPy.assign(state[42+temp_:47+temp_], cardIn4[2:7], 5)

    pIdx = env[83] % 4
    for i in range(4):
        pEnvIdx = (pIdx + i) % 4
        temp1 = 12*i
        temp2 = 15*pEnvIdx

        cuPy.assign(state[201+temp1:213+temp1], env[23+temp2:35+temp2], 12)

        if i == 0:
            for j in range(3):
                cardId = env[35+temp2+j]
                if cardId != -1:
                    cardIn4 = NORMAL_CARD[cardId]
                    temp_ = 11*j
                    state[168+temp_] = cardIn4[0]
                    state[169+temp_+cardIn4[1]] = 1
                    cuPy.assign(state[174+temp_:179+temp_], cardIn4[2:7], 5)

        else:
            temp_ = 3*(i-1)
            for j in range(3):
                cardId = env[35+temp2+j]
                if cardId != -1:
                    if cardId < 40: state[249+temp_] += 1
                    elif cardId < 70: state[250+temp_] += 1
                    else: state[251+temp_] += 1

    cuPy.assign(state[258:263], env[84:89], 5)
    state[263] = env[89]

    if lv1[40] < 40: state[264] = 1
    if lv2[30] < 30: state[265] = 1
    if lv3[20] < 20: state[266] = 1
