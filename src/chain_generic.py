import contextlib
import warnings
import numpy as np
from src import config as C
from src import physics_ball_state, physics_dead_ball, postprocess
from src import pipeline_approach_v13_physics as V13
from src import pipeline_approach_v14_physics as V14
from src import pipeline_approach_v15_physics as V15
HORIZON_S, MARGIN_M, WINDOW, P90_SPEED_THRESHOLD = (0.2, 4.4, 75, 1.842)
_PATCHED = (C, postprocess, physics_ball_state, physics_dead_ball)

@contextlib.contextmanager
def game_constants(halves, pitch):
    saved = [{k: getattr(m, k) for k in ('HALVES', 'PITCH_LENGTH', 'PITCH_WIDTH') if hasattr(m, k)} for m in _PATCHED]
    new = {'HALVES': halves, 'PITCH_LENGTH': pitch[0], 'PITCH_WIDTH': pitch[1]}
    try:
        for m, old in zip(_PATCHED, saved):
            for k in old:
                setattr(m, k, new[k])
        yield
    finally:
        for m, old in zip(_PATCHED, saved):
            for k, v in old.items():
                setattr(m, k, v)

def _velocity(xy):
    v = np.full_like(xy, np.nan)
    v[1:-1] = (xy[2:] - xy[:-2]) * (C.FPS / 2.0)
    return v

def _smooth(codes, window):
    n = len(codes)
    ch = np.concatenate(([0], np.cumsum(codes == 0)))
    ca = np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(n)
    lo, hi = (np.maximum(0, i - window), np.minimum(n, i + window + 1))
    h, a = (ch[hi] - ch[lo], ca[hi] - ca[lo])
    return np.where((h == 0) & (a == 0), 'D', np.where(h >= a, 'H', 'A'))

def base_v6(atd, pitch):
    ball, P, team = (atd['ball'][:, :2], atd['players'][:, :, :2], atd['team'])
    has_ball = ~np.isnan(ball[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        raw = np.linalg.norm(P - ball[:, None, :], axis=2)
        proj = np.linalg.norm(P + _velocity(P) * HORIZON_S - (ball + _velocity(ball) * HORIZON_S)[:, None, :], axis=2)
        proj = np.where(np.isnan(proj), raw, proj)
        best_h = np.nanmin(np.where(team[None] == 0, proj, np.nan), axis=1)
        best_a = np.nanmin(np.where(team[None] == 1, proj, np.nan), axis=1)
        dist = np.where((np.abs(best_h - best_a) < MARGIN_M)[:, None], raw, proj)
        p90 = np.nanpercentile(atd['players'][:, :, 2], 90, axis=1)
    has_player = np.any(~np.isnan(dist), axis=1)
    slot = np.nanargmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
    codes = np.where(has_ball & has_player, team[slot], 2)
    labels = _smooth(codes, WINDOW).astype('<U1')
    x, y = (ball[:, 0], ball[:, 1])
    labels[has_ball & ((x < 0) | (x > pitch[0]) | (y < 0) | (y > pitch[1]))] = 'D'
    labels[~has_ball & (p90 < P90_SPEED_THRESHOLD)] = 'D'
    return labels

def run_chain(atd, halves, pitch, upto='v15'):
    stages = {}
    with game_constants(halves, pitch):
        stages['v6'] = base_v6(atd, pitch)
        stages['v13'], _ = V13.build(stages['v6'], None, atd)
        stages['v14'], _, _ = V14.build(stages['v13'], atd)
        stages['v15'] = V15.build(stages['v14'], atd, verbose=False)
    return stages if upto == 'all' else stages[upto]
