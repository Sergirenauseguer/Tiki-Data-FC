import numpy as np
import pandas as pd
FPS = 25
VMAX, REACT, HORIZON = (7.0, 0.3, 2.0)
TOUCH_MS = 5.0
NEAR_TOUCH, NEAR_CTRL = (2.5, 1.5)
K = 3

def _vel(b, k):
    v = np.full_like(b, np.nan)
    v[k:] = (b[k:] - b[:-k]) * FPS / k
    return v

def _team_nearest(ball, players, team):
    d = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    d = np.where(np.isfinite(d), d, np.inf)
    j = d.argmin(1)
    dmin = d[np.arange(len(d)), j]
    sign = np.where(team[j] == 0, 1.0, -1.0)
    return (dmin, sign)

def _reach(ball, vel, players, team):
    taus = np.arange(0.0, HORIZON + 1e-09, 0.1)
    n = len(ball)
    out = np.full((n, 2), HORIZON + 1.0)
    ok = np.isfinite(ball[:, 0]) & np.isfinite(vel[:, 0])
    idx = np.flatnonzero(ok)
    for c0 in range(0, len(idx), 20000):
        ii = idx[c0:c0 + 20000]
        P = ball[ii, None, :] + vel[ii, None, :] * taus[None, :, None]
        for t, side in ((0, team == 0), (1, team == 1)):
            pl = players[ii][:, side, :2]
            d = np.hypot(pl[:, :, None, 0] - P[:, None, :, 0], pl[:, :, None, 1] - P[:, None, :, 1])
            tt = d / VMAX + REACT
            can = tt <= taus[None, None, :] + 1e-09
            can &= np.isfinite(d)
            first = np.where(can.any(1), taus[None, :], np.inf)
            first = np.where(can.any(1), first, np.inf).min(1)
            out[ii, t] = np.minimum(first, HORIZON + 1.0)
    return out

def _ffill_age(val):
    n = len(val)
    idx = np.where(np.isfinite(val), np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    v = np.where(idx >= 0, val[np.maximum(idx, 0)], 0.0)
    age = np.where(idx >= 0, (np.arange(n) - idx) / FPS, 30.0)
    return (v, np.minimum(age, 30.0))

def _count(mask, lo, hi):
    c = np.r_[0, np.cumsum(mask.astype(np.int32))]
    i = np.arange(len(mask))
    a, b = (np.clip(i + lo, 0, len(mask)), np.clip(i + hi, 0, len(mask)))
    return (c[b] - c[a]).astype(np.float32)

def _half(ball, players, team):
    n = len(ball)
    b = ball[:, :2].astype(float)
    vm, vp = (_vel(b, K), np.full_like(b, np.nan))
    vp[:-K] = vm[K:]
    spm, spp = (np.hypot(*vm.T), np.hypot(*vp.T))
    kink = np.hypot(*(vp - vm).T)
    cosang = np.sum(vm * vp, 1) / np.maximum(spm * spp, 1e-06)
    kturn = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    kratio = spp / np.maximum(spm, 0.5)
    dmin, sign = _team_nearest(b, players, team)
    kink_team = np.where(dmin < NEAR_TOUCH, sign, 0.0)
    near = np.where(dmin < NEAR_CTRL, sign, np.nan)
    near_last, near_since = _ffill_age(near)
    r = near[::-1]
    nl, nu = _ffill_age(r)
    near_until = nu[::-1]
    kk = np.nan_to_num(kink)
    peak = (kk >= TOUCH_MS) & (kk >= np.r_[0, kk[:-1]]) & (kk > np.r_[kk[1:], 0]) & (dmin < NEAR_TOUCH)
    tteam = np.where(peak, sign, np.nan)
    last_t, last_age = _ffill_age(np.r_[np.nan, tteam[:-1]])
    rt = tteam[::-1]
    nx, nx_age = _ffill_age(np.r_[np.nan, rt[:-1]])
    next_t, next_age = (nx[::-1], nx_age[::-1])
    f = {}
    for tag, val in (('h', 1.0), ('a', -1.0)):
        m = peak & (sign == val)
        f['tk_p1_' + tag] = _count(m, -FPS, 0)
        f['tk_p3_' + tag] = _count(m, -3 * FPS, 0)
        f['tk_f1_' + tag] = _count(m, 1, FPS + 1)
        f['tk_f3_' + tag] = _count(m, 1, 3 * FPS + 1)
    tidx = np.flatnonzero(peak)
    chain = np.zeros(len(tidx))
    prev2 = np.zeros(len(tidx))
    for k in range(len(tidx)):
        if k and sign[tidx[k]] == sign[tidx[k - 1]]:
            chain[k] = chain[k - 1] + 1
        else:
            chain[k] = 1
        prev2[k] = sign[tidx[k - 1]] if k else 0.0
    pos = np.searchsorted(tidx, np.arange(n), side='left') - 1
    has = pos >= 0
    f['tk_chain'] = np.where(has, chain[np.maximum(pos, 0)] * 1.0, 0.0)
    f['tk_prev2_ha'] = np.where(has, prev2[np.maximum(pos, 0)], 0.0)
    f['tk_last_ha'], f['tk_last_age'] = (last_t, last_age)
    f['tk_next_ha'], f['tk_next_age'] = (next_t, next_age)
    f['kink'], f['kturn'], f['kratio'], f['kink_ha'] = (kink, kturn, kratio, kink_team)
    f['near_ha'], f['near_since'], f['near_until'] = (near_last, near_since, near_until)
    reach = _reach(b, vm, players, team)
    f['reach_h'], f['reach_a'] = (reach[:, 0], reach[:, 1])
    f['reach_ha'] = reach[:, 1] - reach[:, 0]
    return f

def smooth_ball(ball, sigma):
    from scipy.ndimage import gaussian_filter1d
    if not sigma:
        return ball
    out = ball.copy()
    b = ball[:, :2]
    vis = np.isfinite(b[:, 0])
    st = np.r_[0, np.flatnonzero(vis[1:] != vis[:-1]) + 1]
    en = np.r_[st[1:], len(vis)]
    for a, e in zip(st, en):
        if vis[a] and e - a > 1:
            out[a:e, :2] = gaussian_filter1d(b[a:e].astype(float), sigma, axis=0, mode='nearest')
    return out

def touch_features(atd, halves, pitch=None, sigma=0.0, thr=None):
    global TOUCH_MS
    if thr is not None:
        TOUCH_MS = thr
    ball, players, team = (smooth_ball(atd['ball'], sigma), atd['players'], np.asarray(atd['team']))
    n = len(ball)
    cols = None
    out = None
    for a, b in halves.values():
        f = _half(ball[a:b + 1], players[a:b + 1], team)
        if out is None:
            cols = list(f)
            out = {c: np.zeros(n, np.float32) for c in cols}
        for c in cols:
            out[c][a:b + 1] = np.nan_to_num(np.asarray(f[c], np.float32), nan=0.0)
    return pd.DataFrame(out)
