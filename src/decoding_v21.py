import numpy as np
from src import spell_decoding as SD
FPS = 25
_RUN = None

def _compile():
    import numba

    @numba.njit
    def run(lp, c_ha, c_ah, c_in, c_out):
        n = lp.shape[0]
        back = np.zeros((n, 3), np.int8)
        s = lp[0].copy()
        new = np.empty(3)
        for t in range(1, n):
            a0, a1, a2 = (s[0], s[1] - c_ah[t], s[2] - c_out[t])
            if a0 >= a1 and a0 >= a2:
                new[0], back[t, 0] = (a0, 0)
            elif a1 >= a2:
                new[0], back[t, 0] = (a1, 1)
            else:
                new[0], back[t, 0] = (a2, 2)
            b0, b1, b2 = (s[0] - c_ha[t], s[1], s[2] - c_out[t])
            if b1 >= b0 and b1 >= b2:
                new[1], back[t, 1] = (b1, 1)
            elif b0 >= b2:
                new[1], back[t, 1] = (b0, 0)
            else:
                new[1], back[t, 1] = (b2, 2)
            d0, d1, d2 = (s[0] - c_in[t], s[1] - c_in[t], s[2])
            if d2 >= d0 and d2 >= d1:
                new[2], back[t, 2] = (d2, 2)
            elif d0 >= d1:
                new[2], back[t, 2] = (d0, 0)
            else:
                new[2], back[t, 2] = (d1, 1)
            for j in range(3):
                s[j] = new[j] + lp[t, j]
        path = np.empty(n, np.int8)
        path[n - 1] = np.argmax(s)
        for t in range(n - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path
    return run

def viterbi_tv(p_dead, p_home, c_ha, c_ah, c_dead, eps=0.0001, c_out=None):
    global _RUN
    if _RUN is None:
        _RUN = _compile()
    pdd = np.clip(p_dead, eps, 1 - eps)
    ph = np.clip(p_home, eps, 1 - eps)
    lp = np.column_stack([np.log1p(-pdd) + np.log(ph), np.log1p(-pdd) + np.log1p(-ph), np.log(pdd)])
    n = len(lp)
    c_in = np.broadcast_to(np.asarray(c_dead, np.float64), (n,)).copy()
    c_o = c_in if c_out is None else np.broadcast_to(np.asarray(c_out, np.float64), (n,)).copy()
    path = _RUN(lp.astype(np.float64), np.asarray(c_ha, np.float64), np.asarray(c_ah, np.float64), c_in, c_o)
    return np.array(['H', 'A', 'D'])[path]

def control_candidates(sig, w_s):
    ctrl = np.where(sig['since_ctrl'].values == 0, sig['last_ctrl'].values, np.nan)
    w = int(round(w_s * FPS))
    out = []
    for team in (1.0, -1.0):
        m = (ctrl == team).astype(np.int32)
        c = np.r_[0, np.cumsum(m)]
        i = np.arange(len(m))
        lo, hi = (np.clip(i - w, 0, len(m)), np.clip(i + w + 1, 0, len(m)))
        out.append(c[hi] - c[lo] > 0)
    return out

def rule_pdead(dead_rule, p_rule=0.98):
    return np.where(dead_rule, p_rule, 1 - p_rule)

def masked(probs, d):
    dead, ph, _, sig = probs
    cand_h, cand_a = control_candidates(sig, d['mask_w'])
    gap = sig['ball_vis'].values < 0.5
    extra = d['mask_extra']
    c_ha = d['c_team'] + np.where(cand_a | gap, 0.0, extra)
    c_ah = d['c_team'] + np.where(cand_h | gap, 0.0, extra)
    lab = viterbi_tv(rule_pdead(dead), ph, c_ha, c_ah, d['c_dead'])
    return SD.drop_short(lab, d.get('min_s', 0.0))

def snap(lab, p_change, w_s):
    out = np.asarray(lab).copy()
    st, en = SD.runs(out)
    v = out[st]
    w = int(round(w_s * FPS))
    for k in range(1, len(st)):
        if v[k] == 'D' or v[k - 1] == 'D':
            continue
        b = st[k]
        lo = max(b - w, st[k - 1] + 1)
        hi = min(b + w, en[k] - 1)
        if hi <= lo:
            continue
        nb = lo + int(np.argmax(p_change[lo:hi + 1]))
        if nb < b:
            out[nb:b] = v[k]
        elif nb > b:
            out[b:nb] = v[k - 1]
    return out

def min_gap(lab, ball_vis, min_s, min_s_gap):
    out = np.asarray(lab).copy()
    st, en = SD.runs(out)
    v = out[st]
    for i in range(1, len(st) - 1):
        if v[i] == 'D':
            continue
        L = en[i] - st[i]
        m = min_s_gap if np.mean(ball_vis[st[i]:en[i]] < 0.5) > 0.5 else min_s
        if L >= m * FPS:
            continue
        left, right = (v[i - 1], v[i + 1])
        if left == 'D' and right == 'D':
            out[st[i]:en[i]] = 'D'
        elif left == right and left != 'D' and (left != v[i]):
            out[st[i]:en[i]] = left
    return out

def ball_kick(ball, k=3):
    b = np.asarray(ball)[:, :2].astype(float)
    vm = np.full_like(b, np.nan)
    vm[k:] = (b[k:] - b[:-k]) * FPS / k
    vp = np.full_like(b, np.nan)
    vp[:-k] = vm[k:]
    return np.nan_to_num(np.hypot(*(vp - vm).T), nan=0.0)

def snap_restarts(lab, kick, thr, w_s):
    out = np.asarray(lab).copy()
    st, en = SD.runs(out)
    v = out[st]
    w = int(round(w_s * FPS))
    for k in range(len(st) - 1):
        if v[k] != 'D' or v[k + 1] == 'D':
            continue
        b = en[k]
        lo, hi = (max(st[k] + 1, b - w), min(en[k + 1] - 1, b + w))
        if hi <= lo:
            continue
        cand = np.flatnonzero(kick[lo:hi + 1] >= thr)
        if not len(cand):
            continue
        nb = lo + int(cand[0])
        if nb > b:
            out[b:nb] = 'D'
        elif nb < b:
            out[nb:b] = v[k + 1]
    return out

def snap_restarts_max(lab, kick, thr, w_s):
    out = np.asarray(lab).copy()
    st, en = SD.runs(out)
    v = out[st]
    w = int(round(w_s * FPS))
    for k in range(len(st) - 1):
        if v[k] != 'D' or v[k + 1] == 'D':
            continue
        b = en[k]
        lo, hi = (max(st[k] + 1, b - w), min(en[k + 1] - 1, b + w))
        if hi <= lo:
            continue
        j = lo + int(np.argmax(kick[lo:hi + 1]))
        if kick[j] < thr:
            continue
        if j > b:
            out[b:j] = 'D'
        elif j < b:
            out[j:b] = v[k + 1]
    return out
