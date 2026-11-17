import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
FPS = 25
L, W = (C.PITCH_LENGTH, C.PITCH_WIDTH)
LAG = {1: 4014, 2: 26920}
OUT_M = 1.0
DECEL = 2.0
TRAJ_MAX_S = 3.0
REACH_M = 1.5
KICK_SPEED = 5.0
STILL_SPEED = 1.5
STILL_S = 0.5

def _runs(lab):
    st = np.r_[0, np.flatnonzero(lab[1:] != lab[:-1]) + 1]
    en = np.r_[st[1:], len(lab)]
    return (st, en)

class Match:

    def __init__(self):
        from src.io import load_atd
        from src.postprocess import goalkeeper_slots
        atd = load_atd()
        self.atd = atd
        n = len(atd['ball'])
        self.n = n
        self.halves = dict(C.HALVES)
        ball = np.asarray(atd['ball'], float)[:, :2].copy()
        self.vis_t = np.isfinite(ball[:, 0])
        Bz = np.load(os.path.join(C.CACHE_DIR, 'atd_broadcast.npz'))
        bb = np.asarray(Bz['ball'], float)[:, :2]
        hb = pd.read_csv(os.path.join(os.environ.get('MTAH_DATA_DIR', os.path.join(C.ROOT, 'data')), 'broadcast', 'BAR-ATH_broadcast_homography.csv'), usecols=['frame', 'valid', 'reprojection_error'])
        hv = np.zeros(len(bb), bool)
        ok = hb.frame.values < len(bb)
        hv[hb.frame.values[ok]] = (hb.valid.values[ok] == 1) & (hb.reprojection_error.values[ok] < 5)
        self.replay = np.zeros(n, bool)
        tv = np.full((n, 2), np.nan)
        for half, (a, b) in self.halves.items():
            tt = np.arange(a, b + 1)
            bi = tt - LAG[half]
            k = (bi >= 0) & (bi < len(bb))
            self.replay[tt[k]] = ~hv[bi[k]]
            tv[tt[k]] = np.where(hv[bi[k]][:, None], bb[bi[k]], np.nan)
        fill = ~self.vis_t & np.isfinite(tv[:, 0])
        ball[fill] = tv[fill]
        self.ball = ball
        self.vis = np.isfinite(ball[:, 0])
        sm = pd.DataFrame(ball).rolling(5, center=True, min_periods=3).median().values
        sm[~np.isfinite(ball[:, 0])] = np.nan
        v = np.full((n, 2), np.nan)
        v[4:] = (sm[4:] - sm[:-4]) * (FPS / 4.0)
        self.vel = v
        self.speed = np.hypot(v[:, 0], v[:, 1])
        self.players = np.asarray(atd['players'], float)[:, :, :2]
        self.team = np.asarray(atd['team'])
        pv = np.full(self.players.shape[:2], np.nan)
        pv[1:] = np.hypot(*np.diff(self.players, axis=0).transpose(2, 0, 1)) * FPS
        with np.errstate(all='ignore'):
            self.p90 = np.nanpercentile(pv, 90, axis=1)
        self.p90 = pd.Series(self.p90).rolling(13, center=True, min_periods=1).median().values
        self.gk = goalkeeper_slots(atd)
        self.half_of = np.zeros(n, int)
        for h, (a, b) in self.halves.items():
            self.half_of[a:b + 1] = h

    def inside(self, xy, m=0.0):
        return (xy[..., 0] >= -m) & (xy[..., 0] <= L + m) & (xy[..., 1] >= -m) & (xy[..., 1] <= W + m)

    def nearest(self, f, xy):
        d = np.hypot(self.players[f, :, 0] - xy[0], self.players[f, :, 1] - xy[1])
        if not np.isfinite(d).any():
            return (None, np.inf)
        k = int(np.nanargmin(d))
        return (k, float(d[k]))

    def team_label(self, slot):
        return 'H' if self.team[slot] == 0 else 'A'

def r_out(lab, M):
    out = lab.copy()
    clear = M.vis & ~M.inside(M.ball, OUT_M) & (M.half_of > 0)
    st, en = _runs(clear)
    for a, b in zip(st, en):
        if not clear[a] or b - a < int(0.4 * FPS):
            continue
        s = a
        while s > 0 and M.vis[s - 1] and (not M.inside(M.ball[s - 1])) and (a - s < 3 * FPS):
            s -= 1
        out[s:b] = 'D'
    return out

def predict_path(M, f0, max_s=TRAJ_MAX_S):
    p0, v0 = (M.ball[f0], M.vel[f0])
    if not np.isfinite(v0).all():
        return None
    sp = float(np.hypot(*v0))
    if sp < 4.0:
        return None
    t = np.arange(1, int(max_s * FPS) + 1) / FPS
    t_stop = sp / DECEL
    tt = np.minimum(t, t_stop)
    dist = sp * tt - 0.5 * DECEL * tt ** 2
    return p0[None, :] + (v0 / sp)[None, :] * dist[:, None]

def r_traj_out(lab, M):
    return r_traj(lab, M, parts=('out',))

def r_traj_gk(lab, M):
    return r_traj(lab, M, parts=('gk',))

def r_traj(lab, M, parts=('out', 'gk')):
    out = lab.copy()
    lost = np.flatnonzero(M.vis[:-1] & ~M.vis[1:]) + 1
    for f in lost:
        f0 = f - 1
        if lab[f0] not in ('H', 'A') or M.half_of[f0] == 0:
            continue
        path = predict_path(M, f0)
        if path is None:
            continue
        gap_end = f
        while gap_end < M.n and (not M.vis[gap_end]) and (gap_end - f < 60 * FPS):
            gap_end += 1
        for j, xy in enumerate(path):
            fr = f + j
            if fr >= M.n or fr >= gap_end:
                break
            if not M.inside(xy):
                if 'out' in parts:
                    out[fr:gap_end] = 'D'
                break
            k, d = M.nearest(fr, xy)
            if k is not None and d < REACH_M:
                h = M.half_of[fr]
                is_gk = any((M.gk.get((h, t)) == k for t in (0, 1)))
                near_goal = xy[0] < 16.5 or xy[0] > L - 16.5
                if is_gk and near_goal and ('gk' in parts):
                    out[fr:gap_end] = np.where(out[fr:gap_end] == 'D', M.team_label(k), out[fr:gap_end])
                break
    return out
SPOTS = ('corner', 'penalti', 'puerta', 'centro', 'banda')

def spot_of(xy):
    x, y = xy
    if min((np.hypot(x - cx, y - cy) for cx in (0, L) for cy in (0, W))) < 2.5:
        return 'corner'
    if min((np.hypot(x - px, y - W / 2) for px in (11.0, L - 11.0))) < 1.5:
        return 'penalti'
    if (x < 6.0 or x > L - 6.0) and abs(y - W / 2) < 10.0:
        return 'puerta'
    if np.hypot(x - L / 2, y - W / 2) < 2.0:
        return 'centro'
    if y < 1.5 or y > W - 1.5:
        return 'banda'
    return None

def r_set(lab, M, max_wait_s=60.0):
    out = lab.copy()
    st, en = _runs(lab)
    still_n = int(STILL_S * FPS)
    for a, b in zip(st, en):
        if lab[a] != 'D' or b >= M.n:
            continue
        lo, hi = (a, min(M.n - 1, b + 3 * FPS))
        f = lo
        while f < hi:
            if not (M.vis[f] and M.speed[f] < STILL_SPEED and (spot_of(M.ball[f]) is not None)):
                f += 1
                continue
            g = f
            while g < hi and M.vis[g] and (M.speed[g] < STILL_SPEED):
                g += 1
            if g - f >= still_n and g < hi and (np.nanmax(np.r_[M.speed[g:min(g + FPS // 2, M.n)], 0]) > KICK_SPEED):
                k_slot, d = M.nearest(g, M.ball[f])
                if k_slot is None or d > 3.0:
                    break
                taker = M.team_label(k_slot)
                out[f:g] = 'D'
                e = g
                while e < M.n and out[e] == 'D' and (e - g < 5 * FPS):
                    out[e] = taker
                    e += 1
                e2 = e
                while e2 < M.n and out[e2] not in ('H', 'A'):
                    e2 += 1
                if e2 < M.n and out[e2] != taker:
                    e3 = e2
                    while e3 < M.n and out[e3] == out[e2]:
                        e3 += 1
                    if e3 - e2 < 2 * FPS and e2 - g < 2 * FPS:
                        out[e2:e3] = taker
                break
            f = g + 1
    return out

def r_replay_d(lab, M, min_s=4.0, slow=3.0):
    return r_replay(lab, M, min_s, slow, parts=('d',))

def r_replay_play(lab, M, min_s=4.0, slow=3.0):
    return r_replay(lab, M, min_s, slow, parts=('play',))

def r_replay(lab, M, min_s=4.0, slow=3.0, parts=('d', 'play')):
    out = lab.copy()
    st, en = _runs(M.replay) if 'd' in parts else (np.array([], int), np.array([], int))
    for a, b in zip(st, en):
        if not M.replay[a] or b - a < min_s * FPS or M.half_of[a] == 0:
            continue
        k = np.arange(a, b)
        k = k[(np.nan_to_num(M.p90[k], nan=0.0) < slow) & ~M.vis_t[k]]
        out[k] = 'D'
    if 'play' not in parts:
        return out
    play = ~M.replay & M.vis & (np.nan_to_num(M.speed) > 3.0) & (np.nan_to_num(M.p90) > slow) & M.inside(M.ball)
    st, en = _runs(out)
    for i in range(1, len(st) - 1):
        a, b = (st[i], en[i])
        if out[a] != 'D':
            continue
        pv, nv = (out[a - 1], out[b] if b < M.n else 'D')
        if pv in ('H', 'A'):
            j = a
            while j < b and play[j] and (j - a < 10 * FPS):
                j += 1
            out[a:j] = pv
        if nv in ('H', 'A'):
            j = b
            while j > a and play[j - 1] and (b - j < 10 * FPS) and (out[j - 1] == 'D'):
                j -= 1
            out[j:b] = nv
    return out
TAG_OWN = ('BUILD UP', 'PROGRESSION', 'ATTACKING TRANSITION', 'LONG BALL')
TAG_OPP = ('DEFENDING IN MIDDLE THIRD', 'DEFENDING IN ATTACKING THIRD', 'DEFENSIVE TRANSITION')

def tag_prior(M):
    if hasattr(M, '_tag'):
        return M._tag
    p = os.path.join(C.CACHE_DIR, 'smart_tagging.csv')
    t = pd.read_csv(p)
    prior = np.full(M.n, '', dtype='<U1')
    for _, r in t.iterrows():
        tm = {'Barcelona': 'H', 'Atletic Club': 'A'}.get(r.tag_team)
        if not tm or not 0 <= r.start_frame < r.end_frame:
            continue
        a, b = (int(r.start_frame), int(min(r.end_frame, M.n)))
        if r.code in TAG_OWN:
            prior[a:b] = tm
        elif r.code in TAG_OPP:
            prior[a:b] = 'A' if tm == 'H' else 'H'
    M._tag = prior
    return prior

def r_tag(lab, M, max_s=10.0):
    out = lab.copy()
    prior = tag_prior(M)
    st, en = _runs(lab)
    for i in range(1, len(st) - 1):
        a, b = (st[i], en[i])
        if lab[a] != 'D' or b - a > max_s * FPS:
            continue
        pr = prior[a:b]
        if not (pr != '').all() or len(set(pr)) != 1:
            continue
        if (M.vis[a:b] & ~M.inside(M.ball[a:b], OUT_M)).any() or M.replay[a:b].mean() > 0.5:
            continue
        out[a:b] = pr[0]
    return out
RULES = {'tag': r_tag, 'out': r_out, 'traj': r_traj, 'set': r_set, 'replay': r_replay, 'replay_d': r_replay_d, 'replay_play': r_replay_play, 'traj_out': r_traj_out, 'traj_gk': r_traj_gk}

def apply(lab, M, names):
    for nm in names:
        lab = RULES[nm](lab, M)
    return lab
