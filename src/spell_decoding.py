import numpy as np
FPS = 25

def runs(lab):
    lab = np.asarray(lab)
    st = np.r_[0, np.flatnonzero(lab[1:] != lab[:-1]) + 1]
    en = np.r_[st[1:], len(lab)]
    return (st, en)

def smooth_team(p_home, win_s):
    if win_s <= 0:
        return p_home
    k = max(int(round(win_s * FPS)) | 1, 1)
    z = np.log(np.clip(p_home, 0.0001, 1 - 0.0001) / np.clip(1 - p_home, 0.0001, 1))
    c = np.r_[0.0, np.cumsum(z)]
    h = k // 2
    i = np.arange(len(z))
    lo, hi = (np.clip(i - h, 0, len(z)), np.clip(i + h + 1, 0, len(z)))
    zm = (c[hi] - c[lo]) / (hi - lo)
    return 1 / (1 + np.exp(-zm))

def drop_short(lab, min_s, between_dead=True, inside_team=True):
    if min_s <= 0:
        return lab
    out = np.asarray(lab).copy()
    m = int(round(min_s * FPS))
    for _ in range(5):
        st, en = runs(out)
        v = out[st]
        changed = False
        for i in range(1, len(st) - 1):
            if v[i] == 'D' or en[i] - st[i] >= m:
                continue
            left, right = (v[i - 1], v[i + 1])
            if between_dead and left == 'D' and (right == 'D'):
                out[st[i]:en[i]] = 'D'
                changed = True
            elif inside_team and left == right and (left != 'D') and (left != v[i]):
                out[st[i]:en[i]] = left
                changed = True
        if not changed:
            break
    return out

def decode_rule_dead(decode, dead_rule, p_home, c_team, c_dead, p_rule=0.98, smooth_s=0.0, min_s=0.0, p_dead_learned=None, w_rule=1.0):
    p_dead = np.where(dead_rule, p_rule, 1 - p_rule)
    if p_dead_learned is not None and w_rule < 1:
        p_dead = w_rule * p_dead + (1 - w_rule) * p_dead_learned
    lab = decode(p_dead, smooth_team(p_home, smooth_s), c_team, c_dead)
    return drop_short(lab, min_s)
