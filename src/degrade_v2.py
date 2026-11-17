import warnings
import numpy as np
from src.degrade import _simulate, add_jumps, carry_dead_ball
FPS = 25
PLAYER_GAP_S = 5.0
BOX_X, BOX_HALF_W = (16.5, 20.16)
CROWD_R = 5.0
SPEED_EDGES = (0.0, 5.0, 12.0, 1000000000.0)
CROWD_EDGES = (0, 1, 3, 99)
N_SPEED, N_CROWD = (len(SPEED_EDGES) - 1, len(CROWD_EDGES) - 1)
N_CTX = 2 * N_SPEED * N_CROWD
MIN_COUNT = 2000

def _runs(mask):
    mask = np.asarray(mask, bool)
    st = np.r_[0, np.flatnonzero(mask[1:] != mask[:-1]) + 1]
    en = np.r_[st[1:], len(mask)]
    return [(a, b) for a, b in zip(st, en) if mask[a]]

def context(ball, players, pitch):
    L, W = pitch
    x, y = (ball[:, 0], ball[:, 1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        in_box = ((x < BOX_X) | (x > L - BOX_X)) & (np.abs(y - W / 2) < BOX_HALF_W)
        d = np.hypot(players[:, :, 0] - x[:, None], players[:, :, 1] - y[:, None])
        crowd = np.sum(d < CROWD_R, axis=1)
        v = np.full(len(x), np.nan)
        v[4:] = np.hypot(x[4:] - x[:-4], y[4:] - y[:-4]) * FPS / 4
    v = np.where(np.isnan(v), 0.0, v)
    sb = np.clip(np.searchsorted(SPEED_EDGES, v, side='right') - 1, 0, N_SPEED - 1)
    cb = np.clip(np.searchsorted(CROWD_EDGES, crowd, side='right') - 1, 0, N_CROWD - 1)
    ctx = (in_box.astype(int) * N_SPEED + sb) * N_CROWD + cb
    return np.where(np.isnan(x), -1, ctx)

def fit_atd(atd, state_labels, graded, pitch):
    dead = np.asarray(state_labels) == 'D'
    graded = np.asarray(graded, bool)
    ball = atd['ball']
    vis = ~np.isnan(ball[:, 0])
    n = len(dead)
    stoppages = []
    for a, b in _runs(dead & graded):
        if a > 0 and b < n and graded[a - 1] and graded[min(b, n - 1)]:
            stoppages.append(vis[a:b].copy())
    ctx = context(ball, atd['players'], pitch)
    live = ~dead & graded
    prev_vis = np.r_[False, vis[:-1]]
    prev_ctx = np.r_[-1, ctx[:-1]]
    at_risk = live & prev_vis & np.r_[False, live[:-1]]
    lost = at_risk & ~vis
    hazard = np.zeros(N_CTX)
    count = np.zeros(N_CTX, int)
    for k in range(N_CTX):
        m = at_risk & (prev_ctx == k)
        count[k] = m.sum()
        hazard[k] = lost[m].sum() / max(m.sum(), 1)
    for box in (0, 1):
        sel = np.arange(N_CTX) // (N_SPEED * N_CROWD) == box
        m = at_risk & np.isin(prev_ctx, np.flatnonzero(sel))
        pooled = lost[m].sum() / max(m.sum(), 1)
        hazard[sel & (count < MIN_COUNT)] = pooled
    gap_len = {0: [], 1: []}
    for a, b in _runs(~vis & graded):
        if a > 0 and live[a] and vis[a - 1] and (prev_ctx[a] >= 0):
            if b < n and dead[a:b].mean() <= 0.5:
                gap_len[int(prev_ctx[a] // (N_SPEED * N_CROWD))].append(b - a)
    present = (~np.isnan(atd['players'][:, :, 0])).sum(axis=1) / 22.0
    p_gain = 1.0 / (PLAYER_GAP_S * FPS)
    players = np.zeros((2, 2))
    for k, m in ((0, ~dead & graded), (1, dead & graded)):
        f = min(present[m].mean(), 0.999)
        players[k] = (p_gain * (1 - f) / f, p_gain)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        dd = np.nanmin(np.hypot(atd['players'][:, :, 0] - ball[:, None, 0], atd['players'][:, :, 1] - ball[:, None, 1]), axis=1)
    sel = dead & graded & vis & np.isfinite(dd)
    keep = np.percentile(dd[sel], np.arange(5, 100, 10)).tolist()
    b2 = ball[:, :2]
    v = np.full(n, np.nan)
    v[2:-2] = np.hypot(*(b2[4:] - b2[:-4]).T) * FPS / 4
    selv = dead & graded & np.isfinite(v)
    speed = np.percentile(v[selv], np.arange(5, 100, 10)).tolist()
    jump = np.r_[np.hypot(*(b2[1:] - b2[:-1]).T), np.nan]
    okj = np.isfinite(jump) & graded & np.r_[graded[1:], False]
    jump_rate = [float(np.mean(jump[okj & (dead == bool(k))] > 1.0)) for k in (0, 1)]
    jump_size = np.percentile(jump[okj & (jump > 1.0)], np.arange(5, 100, 10)).tolist()
    return {'stoppages': stoppages, 'stop_len': np.array([len(s) for s in stoppages]), 'hazard': hazard, 'hazard_count': count, 'gap_len': {k: np.array(v) for k, v in gap_len.items()}, 'players': players, 'jump_rate': jump_rate, 'jump_size': jump_size, 'dead_keep_dist': keep, 'dead_ball_speed': speed, 'ball_visible': [float(vis[~dead & graded].mean()), float(vis[dead & graded].mean())]}

def _transplant(src, length, rng):
    lb = len(src)
    h1 = length // 2
    h2 = length - h1
    if lb >= length:
        return np.r_[src[:h1], src[lb - h2:]]
    mid = src[lb // 4:lb - lb // 4] if lb >= 8 else src
    fill = length - lb
    reps = int(np.ceil(fill / max(len(mid), 1)))
    middle = np.tile(mid, reps)[:fill]
    return np.r_[src[:lb // 2], middle, src[lb // 2:]]

def ball_visibility(state, ball, players, model, pitch, rng):
    n = len(state)
    dead = state.astype(bool)
    vis = np.ones(n, bool)
    logs = np.log(model['stop_len'])
    for a, b in _runs(dead):
        k = np.argsort(np.abs(logs - np.log(b - a)))[:5]
        vis[a:b] = _transplant(model['stoppages'][int(rng.choice(k))], b - a, rng)
    ctx = context(ball, players, pitch)
    hz = model['hazard']
    u = rng.random(n)
    t = 0
    while t < n:
        if dead[t]:
            t += 1
            continue
        c = ctx[t - 1] if t > 0 else -1
        if t > 0 and vis[t - 1] and (c >= 0) and (u[t] < hz[c]):
            box = int(c // (N_SPEED * N_CROWD))
            pool = model['gap_len'][box] if len(model['gap_len'][box]) else model['gap_len'][1 - box]
            L = int(rng.choice(pool))
            e = min(t + L, n)
            seg = ~dead[t:e]
            vis[t:e][seg] = False
            t = e
        else:
            t += 1
    return vis

def degrade(match, model, seed=0, pitch=(105.0, 68.0)):
    rng = np.random.default_rng(seed)
    state = (match['label'] == 'D').astype(int)
    out = dict(match)
    players = match['players'].copy()
    players[~_simulate(state, model['players'], players.shape[1], rng)] = np.nan
    ball = carry_dead_ball(match['ball'], match['players'], state.astype(bool), rng, model['dead_keep_dist'], model['dead_ball_speed'])
    ball = add_jumps(ball, match['ball'], state, model, rng)
    vis = ball_visibility(state, ball, players, model, pitch, rng)
    ball[~vis] = np.nan
    ball[:, 2] = np.nan
    out['ball'], out['players'] = (ball, players)
    return out
BALL_SMOOTH_SIGMA = 6.0

def degrade_v3(match, model, seed=0, pitch=(105.0, 68.0)):
    from src.touch_features import smooth_ball
    rng = np.random.default_rng(seed)
    state = (match['label'] == 'D').astype(int)
    out = dict(match)
    players = match['players'].copy()
    players[~_simulate(state, model['players'], players.shape[1], rng)] = np.nan
    ball = carry_dead_ball(match['ball'], match['players'], state.astype(bool), rng, model['dead_keep_dist'], model['dead_ball_speed'])
    ball = smooth_ball(ball, BALL_SMOOTH_SIGMA)
    ball = add_jumps(ball, match['ball'], state, model, rng)
    vis = ball_visibility(state, ball, players, model, pitch, rng)
    ball[~vis] = np.nan
    ball[:, 2] = np.nan
    out['ball'], out['players'] = (ball, players)
    return out
DRIFT_AMP_M, DRIFT_TAU_S = (1.0, 2.0)

def _drift(n, rng):
    from scipy.ndimage import gaussian_filter1d
    z = gaussian_filter1d(rng.normal(size=(n, 2)), DRIFT_TAU_S * FPS, axis=0)
    z /= z.std(0)
    return z * DRIFT_AMP_M / np.sqrt(2)

def degrade_v45(match, model, seed=0, pitch=(105.0, 68.0), smooth=False):
    from src.touch_features import smooth_ball
    rng = np.random.default_rng(seed)
    state = (match['label'] == 'D').astype(int)
    out = dict(match)
    players = match['players'].copy()
    players[~_simulate(state, model['players'], players.shape[1], rng)] = np.nan
    ball = carry_dead_ball(match['ball'], match['players'], state.astype(bool), rng, model['dead_keep_dist'], model['dead_ball_speed'])
    if smooth:
        ball = smooth_ball(ball, BALL_SMOOTH_SIGMA)
    ball = add_jumps(ball, match['ball'], state, model, rng)
    vis = ball_visibility(state, ball, players, model, pitch, rng)
    ball[:, :2] = ball[:, :2] + _drift(len(ball), np.random.default_rng(10000 + seed))
    ball[~vis] = np.nan
    ball[:, 2] = np.nan
    out['ball'], out['players'] = (ball, players)
    return out
BLACKOUT_MIN_TEAM = 8

def fit_blackouts(atd, rules, graded):
    pl = np.asarray(atd['players'], float)
    team = np.asarray(atd['team'])
    ok = np.isfinite(pl[:, :, 0])
    mn = np.minimum(ok[:, team == 0].sum(1), ok[:, team == 1].sum(1))
    few = (mn < BLACKOUT_MIN_TEAM) & graded
    dead = (np.asarray(rules) == 'D') & graded
    st = np.r_[0, np.flatnonzero(few[1:] != few[:-1]) + 1]
    en = np.r_[st[1:], len(few)]
    lens = np.array([b - a for a, b in zip(st, en) if few[a] and dead[a:b].mean() > 0.5])
    rate = len(lens) / max(dead.sum() / (25 * 60), 1e-06)
    frac = float(ok[few].sum() / max(few.sum() * 22, 1))
    return {'rate_per_dead_min': float(rate), 'lens': lens.astype(int), 'visible_frac': frac}

def degrade_v6(match, model, seed=0, pitch=(105.0, 68.0)):
    out = degrade_v45(match, model, seed, pitch, smooth=True)
    bo = model['blackout']
    rng = np.random.default_rng(20000 + seed)
    players = out['players'].copy()
    dead = np.asarray(match['label']) == 'D'
    st = np.r_[0, np.flatnonzero(dead[1:] != dead[:-1]) + 1]
    en = np.r_[st[1:], len(dead)]
    for a, b in zip(st, en):
        if not dead[a]:
            continue
        k = rng.poisson(bo['rate_per_dead_min'] * (b - a) / (25 * 60))
        for _ in range(k):
            L = int(min(rng.choice(bo['lens']), b - a))
            s0 = a + int(rng.integers(0, b - a - L + 1))
            hide = rng.random(players.shape[1]) > bo['visible_frac']
            players[s0:s0 + L, hide] = np.nan
    out['players'] = players
    return out
