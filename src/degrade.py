import numpy as np
PLAYER_GAP_S = 5.0
FPS = 25
CARRY_MAX_SPEED = 6.0
CARRY_RADIUS_M = 20.0
PLACE_S = 2.0
NOISE_M = 0.15
PROFILE_EDGES_S = (0, 1, 2, 4, 8, 20, 1000000000.0)

def _dist_to_edge(dead):
    n = len(dead)
    out = np.full(n, np.nan)
    st = np.r_[0, np.flatnonzero(dead[1:] != dead[:-1]) + 1]
    en = np.r_[st[1:], n]
    for a, b in zip(st, en):
        if dead[a]:
            k = np.arange(b - a)
            out[a:b] = np.minimum(k, k[::-1]) / FPS
    return out

def _bin(dist):
    return np.clip(np.searchsorted(PROFILE_EDGES_S, np.nan_to_num(dist), side='right') - 1, 0, len(PROFILE_EDGES_S) - 2)

def _rates(visible, dead):
    prev, cur, st = (visible[:-1], visible[1:], dead[1:])
    out = np.zeros((2, 2))
    for k in (0, 1):
        m = st == k
        pv, cv = (prev[m], cur[m])
        out[k] = ((pv & ~cv).sum() / max(pv.sum(), 1), (~pv & cv).sum() / max((~pv).sum(), 1))
    return out

def fit_visibility(atd, state_labels):
    dead = (np.asarray(state_labels) == 'D').astype(int)
    ball = ~np.isnan(atd['ball'][:, 0])
    present = (~np.isnan(atd['players'][:, :, 0])).sum(axis=1) / 22.0
    p_gain = 1.0 / (PLAYER_GAP_S * FPS)
    players = np.zeros((2, 2))
    for k in (0, 1):
        f = min(present[dead == k].mean(), 0.999)
        players[k] = (p_gain * (1 - f) / f, p_gain)
    edge = _dist_to_edge(dead.astype(bool))
    b = _bin(edge)
    profile = [float(ball[(dead == 1) & (b == k)].mean()) for k in range(len(PROFILE_EDGES_S) - 1)]
    with np.errstate(invalid='ignore'):
        dd = np.nanmin(np.hypot(atd['players'][:, :, 0] - atd['ball'][:, None, 0], atd['players'][:, :, 1] - atd['ball'][:, None, 1]), axis=1)
    sel = (dead == 1) & ball & np.isfinite(dd)
    keep = np.percentile(dd[sel], np.arange(5, 100, 10)).tolist()
    b = atd['ball'][:, :2]
    v = np.full(len(b), np.nan)
    v[2:-2] = np.hypot(*(b[4:] - b[:-4]).T) * FPS / 4
    selv = (dead == 1) & np.isfinite(v)
    speed = np.percentile(v[selv], np.arange(5, 100, 10)).tolist()
    jump = np.r_[np.hypot(*(b[1:] - b[:-1]).T), np.nan]
    okj = np.isfinite(jump)
    jump_rate = [float(np.mean(jump[okj & (dead == k)] > 1.0)) for k in (0, 1)]
    jump_size = np.percentile(jump[okj & (jump > 1.0)], np.arange(5, 100, 10)).tolist()
    return {'ball': _rates(ball, dead), 'players': players, 'dead_profile': profile, 'jump_rate': jump_rate, 'jump_size': jump_size, 'dead_keep_dist': keep, 'dead_ball_speed': speed, 'ball_visible': [float(ball[dead == k].mean()) for k in (0, 1)], 'players_present': [float(present[dead == k].mean()) for k in (0, 1)]}

def _simulate(state, tr, n_series, rng):
    n = len(state)
    u = rng.random((n, n_series))
    vis = np.empty((n, n_series), bool)
    stationary = tr[:, 1] / (tr[:, 0] + tr[:, 1])
    cur = u[0] < stationary[state[0]]
    p_lose, p_gain = (tr[state, 0], tr[state, 1])
    change = np.r_[False, state[1:] != state[:-1]]
    for i in range(n):
        if change[i]:
            cur = u[i] < stationary[state[i]]
        else:
            cur = np.where(cur, u[i] >= p_lose[i], u[i] < p_gain[i])
        vis[i] = cur
    return vis

def _simulate_ball(state, tr, profile, rng):
    n = len(state)
    dead = state.astype(bool)
    pi = np.where(dead, np.asarray(profile)[_bin(_dist_to_edge(dead))], tr[0, 1] / tr[0].sum())
    r = np.where(dead, tr[1].sum(), tr[0].sum())
    p_gain, p_lose = (r * pi, r * (1 - pi))
    u = rng.random(n)
    vis = np.empty(n, bool)
    change = np.r_[True, state[1:] != state[:-1]]
    cur = False
    for i in range(n):
        cur = u[i] < pi[i] if change[i] else u[i] >= p_lose[i] if cur else u[i] < p_gain[i]
        vis[i] = cur
    return vis
KEEP_RESAMPLE_S = 3.0
PASS_PROB = 0.3
PASS_RADIUS_M = 25.0

def carry_dead_ball(ball, players, dead, rng, keep_dist=None, speeds=None):
    out = ball.copy()
    P = players[:, :, :2]
    n = len(dead)
    st = np.r_[0, np.flatnonzero(dead[1:] != dead[:-1]) + 1]
    en = np.r_[st[1:], n]
    step = CARRY_MAX_SPEED / FPS
    for a, b in zip(st, en):
        if not dead[a]:
            continue
        p0 = ball[a, :2] if np.isfinite(ball[a, 0]) else ball[a - 1, :2] if a > 0 else None
        if p0 is None or not np.isfinite(p0).all():
            continue
        restart = ball[b, :2] if b < n and np.isfinite(ball[b, 0]) else p0
        pos = p0.astype(float).copy()
        place_from = b - int(PLACE_S * FPS)
        keep = 0.0
        holder, vmax = (-1, CARRY_MAX_SPEED)
        for t in range(a, b):
            if keep_dist is not None and (t - a) % int(KEEP_RESAMPLE_S * FPS) == 0:
                keep = float(rng.choice(keep_dist))
                vmax = max(float(rng.choice(speeds)), 0.5) if speeds is not None else CARRY_MAX_SPEED
                holder = -1
                if speeds is not None and rng.random() < PASS_PROB:
                    dp = np.hypot(P[t, :, 0] - pos[0], P[t, :, 1] - pos[1])
                    cand = np.flatnonzero(np.isfinite(dp) & (dp > 3.0) & (dp < PASS_RADIUS_M))
                    if len(cand):
                        holder = int(rng.choice(cand))
            if t >= place_from:
                target = restart
            elif holder >= 0 and np.isfinite(P[t, holder, 0]):
                target = P[t, holder]
            else:
                d = np.hypot(P[t, :, 0] - p0[0], P[t, :, 1] - p0[1])
                j = np.nanargmin(np.where(np.isfinite(d), d, np.inf)) if np.isfinite(d).any() else -1
                target = P[t, j] if j >= 0 and d[j] < CARRY_RADIUS_M else p0
                gap = pos - target
                g = float(np.hypot(*gap))
                if keep > 0 and np.isfinite(g):
                    target = target + (gap / g if g > 1e-06 else rng.normal(0, 1, 2)) * keep
            delta = target - pos
            dist = float(np.hypot(*delta))
            step = vmax / FPS
            if dist > step:
                delta = delta * (step / dist)
            pos = pos + delta
            out[t, :2] = pos + rng.normal(0, NOISE_M, 2)
    return out

def add_jumps(ball, clean_ball, state, model, rng):
    n = len(state)
    j = np.r_[np.hypot(*(clean_ball[1:, :2] - clean_ball[:-1, :2]).T), np.nan]
    ok = np.isfinite(j)
    own = [float(np.mean(j[ok & (state == k)] > 1.0)) if np.any(ok & (state == k)) else 0.0 for k in (0, 1)]
    extra = np.array([max(model['jump_rate'][k] - own[k], 0.0) for k in (0, 1)])
    hit = rng.random(n) < extra[state]
    out = ball.copy()
    off = np.zeros(2)
    decay = np.exp(-1.0 / (0.5 * FPS))
    for t in range(n):
        if hit[t]:
            ang = rng.uniform(0, 2 * np.pi)
            off = float(rng.choice(model['jump_size'])) * np.array([np.cos(ang), np.sin(ang)])
        elif state[t] == 0:
            off = off * decay
        out[t, :2] = out[t, :2] + off
    return out

def degrade(match, model, seed=0):
    rng = np.random.default_rng(seed)
    state = (match['label'] == 'D').astype(int)
    out = dict(match)
    ball = carry_dead_ball(match['ball'], match['players'], state.astype(bool), rng, model.get('dead_keep_dist'), model.get('dead_ball_speed'))
    if 'jump_rate' in model:
        ball = add_jumps(ball, match['ball'], state, model, rng)
    if 'dead_profile' in model:
        ball[~_simulate_ball(state, model['ball'], model['dead_profile'], rng)] = np.nan
    else:
        ball[~_simulate(state, model['ball'], 1, rng)[:, 0]] = np.nan
    ball[:, 2] = np.nan
    players = match['players'].copy()
    players[~_simulate(state, model['players'], players.shape[1], rng)] = np.nan
    out['ball'], out['players'] = (ball, players)
    return out
