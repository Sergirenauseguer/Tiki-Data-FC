import warnings
import numpy as np
from src.config import HALVES, HOME, AWAY, N_FRAMES, PITCH_LENGTH, PITCH_WIDTH

def graded_mask(n=N_FRAMES):
    mask = np.zeros(n, dtype=bool)
    for first, last in HALVES.values():
        mask[first:last + 1] = True
    return mask

def half_of_frames(n=N_FRAMES):
    out = np.zeros(n, dtype=np.int8)
    for half, (first, last) in HALVES.items():
        out[first:last + 1] = half
    return out

def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

def label_runs(labels, valid):
    out = []
    for start, stop in runs(valid):
        window = labels[start:stop]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        out += [(start + a, start + b, window[a]) for a, b in zip(bounds[:-1], bounds[1:])]
    return out

def ball_visible(atd):
    return ~np.isnan(atd['ball'][:, 0])

def ball_out_of_pitch(atd, margin=0.0):
    x, y = (atd['ball'][:, 0], atd['ball'][:, 1])
    with np.errstate(invalid='ignore'):
        return (x < -margin) | (x > PITCH_LENGTH + margin) | (y < -margin) | (y > PITCH_WIDTH + margin)

def ball_step_speed(atd, fps=25):
    xy = atd['ball'][:, :2]
    step = np.full(len(xy), np.nan, dtype=np.float32)
    step[1:] = np.hypot(*(xy[1:] - xy[:-1]).T) * fps
    return step

def player_ball_distance(atd):
    players, ball = (atd['players'], atd['ball'])
    return np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])

def nearest_by_team(atd):
    distance = player_ball_distance(atd)
    team = atd['team']
    filled = np.where(np.isnan(distance), np.inf, distance)
    d_home = filled[:, team == HOME].min(axis=1)
    d_away = filled[:, team == AWAY].min(axis=1)
    slot = np.argmin(filled, axis=1)
    nearest_team = team[slot].astype(np.int8)
    missing = np.isinf(np.minimum(d_home, d_away))
    nearest_team[missing] = -1
    slot[missing] = -1
    d_home[np.isinf(d_home)] = np.nan
    d_away[np.isinf(d_away)] = np.nan
    return (d_home, d_away, slot, nearest_team)

def players_per_team(atd):
    present = ~np.isnan(atd['players'][:, :, 0])
    team = atd['team']
    return (present[:, team == HOME].sum(axis=1), present[:, team == AWAY].sum(axis=1))

def mean_speed(atd, team_id=None):
    speed = atd['players'][:, :, 2]
    if team_id is not None:
        speed = speed[:, atd['team'] == team_id]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(speed, axis=1)

def team_centroid(atd, team_id):
    xy = atd['players'][:, atd['team'] == team_id, :2]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(xy, axis=1)

def attack_direction(atd):
    half = half_of_frames(len(atd['ball']))
    out = {}
    for h in HALVES:
        cx_home = np.nanmean(team_centroid(atd, HOME)[half == h, 0])
        cx_away = np.nanmean(team_centroid(atd, AWAY)[half == h, 0])
        out[h] = {HOME: 1 if cx_home < cx_away else -1, AWAY: 1 if cx_away < cx_home else -1}
    return out

def auc(score, positive):
    keep = ~np.isnan(score)
    score, positive = (score[keep], positive[keep].astype(bool))
    n_pos, n_neg = (positive.sum(), (~positive).sum())
    if not n_pos or not n_neg:
        return np.nan
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    sorted_scores = score[order]
    _, first, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    for f, c in zip(first[counts > 1], counts[counts > 1]):
        ranks[order[f:f + c]] = f + (c + 1) / 2.0
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
