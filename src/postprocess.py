import numpy as np
from src.config import FPS, HALVES

def runs_of(labels, start, end):
    seg = labels[start:end]
    if len(seg) == 0:
        return []
    change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(seg)]))
    return [(start + bounds[i], start + bounds[i + 1], seg[bounds[i]]) for i in range(len(bounds) - 1)]
KICKOFF_CENTRE_M = 3.0

def detect_half_kickoffs(tagging, ball_xy):
    from src.config import PITCH_LENGTH, PITCH_WIDTH
    x, y = (ball_xy[:, 0], ball_xy[:, 1])
    with np.errstate(invalid='ignore'):
        dc = np.hypot(x - PITCH_LENGTH / 2, y - PITCH_WIDTH / 2)
    events = [e for e in tagging if e['code'].strip().upper() == 'SET PIECES' and 'kick off (start)' in str(e['tags'].get('Type', '')).lower()]
    out = {}
    for half, (a, b) in HALVES.items():
        ev = [e for e in events if a - 10 * FPS <= e['start_frame'] <= b]
        if not ev:
            continue
        e = min(ev, key=lambda e: e['start_frame'])
        lo, hi = (max(a, e['start_frame']), min(b + 1, e['end_frame'] + 2 * FPS))
        w = dc[lo:hi]
        if np.isfinite(w).any() and np.nanmin(w) < KICKOFF_CENTRE_M:
            out[half] = lo + int(np.nanargmin(w))
        else:
            out[half] = (lo + hi) // 2
    return out

def force_pre_kickoff_dead(labels, kickoffs):
    out = labels.copy()
    for half, k in kickoffs.items():
        a = HALVES[half][0]
        if k > a:
            out[a:k] = 'D'
    return out

def prune_blips(labels, max_team_s, max_dead_s):
    out = labels.copy()
    max_team = int(round(max_team_s * FPS))
    max_dead = int(round(max_dead_s * FPS))
    for start, end in ((a, b + 1) for a, b in HALVES.values()):
        while True:
            runs = runs_of(out, start, end)
            cands = []
            for i in range(1, len(runs) - 1):
                (_, _, prev), (s, e, cur), (_, _, nxt) = (runs[i - 1], runs[i], runs[i + 1])
                if prev != nxt or prev not in ('H', 'A'):
                    continue
                n = e - s
                if cur in ('H', 'A') and cur != prev and (n < max_team) or (cur == 'D' and n < max_dead):
                    cands.append((n, s, e, prev))
            if not cands:
                break
            cands.sort()
            touched = np.zeros(end - start, dtype=bool)
            for n, s, e, fill in cands:
                lo, hi = (max(s - 1, start), min(e + 1, end))
                if touched[lo - start:hi - start].any():
                    continue
                out[s:e] = fill
                touched[lo - start:hi - start] = True
    return out
GOAL_MOUTH_HALF = 3.66 + 3.0
CENTRE_RADIUS = 3.0
KICK_SPEED = 2.0
NEAR_GOAL_LINE = 6.0

def goal_to_kickoff_dead(labels, tagging, ball_xy):
    from src.config import PITCH_LENGTH, PITCH_WIDTH
    out = labels.copy()
    x, y = (ball_xy[:, 0], ball_xy[:, 1])
    step = np.full(len(x), np.nan)
    step[1:] = np.hypot(np.diff(x), np.diff(y)) * FPS
    goals = [e for e in tagging if e['code'].strip().upper() == 'GOALS']
    kos = sorted((e for e in tagging if e['code'].strip().upper() == 'SET PIECES' and 'after goal' in str(e['tags'].get('Type', '')).lower()), key=lambda e: e['start_frame'])
    info = []
    with np.errstate(invalid='ignore'):
        in_goal = ((x < 0) | (x > PITCH_LENGTH)) & (np.abs(y - PITCH_WIDTH / 2) < GOAL_MOUTH_HALF)
        near_goal = (np.minimum(x, PITCH_LENGTH - x) < NEAR_GOAL_LINE) & (np.abs(y - PITCH_WIDTH / 2) < GOAL_MOUTH_HALF)
        at_centre = np.hypot(x - PITCH_LENGTH / 2, y - PITCH_WIDTH / 2) < CENTRE_RADIUS
        moving = step > KICK_SPEED
    for g in goals:
        ko = next((k for k in kos if k['end_frame'] > g['start_frame']), None)
        if ko is None:
            info.append('sin saque')
            continue
        hits = np.flatnonzero(in_goal[g['start_frame']:ko['start_frame']])
        start_how = 'en la portería'
        if len(hits) == 0:
            hits = np.flatnonzero(near_goal[g['start_frame']:ko['start_frame']])
            start_how = 'cerca de la portería'
        if len(hits) == 0:
            info.append('balón no visto cerca de la portería')
            continue
        d0 = g['start_frame'] + hits[-1]
        w0, w1 = (ko['start_frame'], ko['end_frame'] + 2 * FPS)
        kick = np.flatnonzero(at_centre[w0:w1] & moving[w0:w1])
        d1, how = (w0 + kick[0], 'saque visto') if len(kick) else ((ko['start_frame'] + ko['end_frame']) // 2, 'saque estimado')
        if not 10 * FPS <= d1 - d0 <= 300 * FPS:
            info.append('duración implausible')
            continue
        out[d0:d1] = 'D'
        info.append('aplicado (balón visto {}, {})'.format(start_how, how))
    return (out, info)

def backdate_debounce(labels, min_run, min_dead_run):
    out = labels.copy()
    n = len(labels)
    runs = runs_of(labels, 0, n)
    starts = []
    for s, _, lab in runs:
        shift = (min_dead_run if lab == 'D' else min_run) - 1 if s > 0 else 0
        starts.append(max(0, s - max(shift, 0)))
    keep = []
    for i in range(len(runs) - 1, -1, -1):
        if not keep or starts[i] < starts[keep[-1]]:
            keep.append(i)
    keep.reverse()
    for j, i in enumerate(keep):
        s = starts[i] if j else 0
        e = starts[keep[j + 1]] if j + 1 < len(keep) else n
        out[s:e] = runs[i][2]
    return out

def dead_by_player_speed(labels, p90_speed, window_s=2.0, max_speed=2.0, min_run_s=3.0):
    n = int(round(window_s * FPS))
    with np.errstate(invalid='ignore'):
        x = np.where(np.isfinite(p90_speed), p90_speed, np.nan)
    import pandas as pd
    smooth = pd.Series(x).rolling(2 * n + 1, center=True, min_periods=1).mean().values
    low = np.r_[0, (smooth < max_speed).astype(np.int8), 0]
    starts = np.flatnonzero(np.diff(low) == 1)
    ends = np.flatnonzero(np.diff(low) == -1)
    out = labels.copy()
    min_run = int(round(min_run_s * FPS))
    for s, e in zip(starts, ends):
        if e - s >= min_run:
            out[s:e] = np.where(out[s:e] == 'X', 'X', 'D')
    return out

def require_control(labels, dist_home, dist_away, ball_visible, radius_m=1.5, min_s=0.4, min_visible=0.5):
    out = labels.copy()
    need = int(round(min_s * FPS))
    for a, b in HALVES.values():
        runs = runs_of(out, a, b + 1)
        for i in range(1, len(runs)):
            prev, (s, e, cur) = (runs[i - 1][2], runs[i])
            if cur not in ('H', 'A') or prev not in ('H', 'A') or prev == cur:
                continue
            if ball_visible[s:e].mean() < min_visible:
                continue
            d = (dist_home if cur == 'H' else dist_away)[s:e]
            with np.errstate(invalid='ignore'):
                close = np.r_[0, (d < radius_m).astype(np.int8), 0]
            st, en = (np.flatnonzero(np.diff(close) == 1), np.flatnonzero(np.diff(close) == -1))
            if (en - st).max(initial=0) < need:
                out[s:e] = prev
    return out

def goalkeeper_slots(atd):
    import warnings
    from src.config import AWAY, HOME, PITCH_LENGTH
    players, team = (atd['players'], atd['team'])
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for half, (a, b) in HALVES.items():
            for t in (HOME, AWAY):
                slots = np.flatnonzero(team == t)
                x = players[a:b + 1, slots, 0]
                present = np.isfinite(x).mean(axis=0) > 0.5
                mean_x = np.nanmean(x, axis=0)
                own_goal = 0.0 if np.nanmean(mean_x[present]) < PITCH_LENGTH / 2 else PITCH_LENGTH
                gap = np.where(present, np.abs(mean_x - own_goal), np.inf)
                out[half, t] = int(slots[np.argmin(gap)])
    return out

def keeper_hold_alive(labels, atd, gk_slots, radius_m=1.5, look_s=3.0, max_hold_s=15.0):
    from src.config import AWAY, HOME
    out = labels.copy()
    ball = atd['ball'][:, :2]
    look = int(round(look_s * FPS))
    max_hold = int(round(max_hold_s * FPS))
    code = {'H': HOME, 'A': AWAY}
    n_fixed = 0
    for half, (a, b) in HALVES.items():
        runs = runs_of(labels, a, b + 1)
        for i in range(1, len(runs)):
            prev, (s, e, cur) = (runs[i - 1][2], runs[i])
            if cur != 'D' or prev not in ('H', 'A') or e - s > max_hold:
                continue
            gk = atd['players'][s:min(e, s + look), gk_slots[half, code[prev]], :2]
            with np.errstate(invalid='ignore'):
                d = np.hypot(*(gk - ball[s:min(e, s + look)]).T)
            if np.isfinite(d).any() and np.nanmin(d) < radius_m:
                out[s:e] = prev
                n_fixed += 1
    return (out, n_fixed)

def delay_team_changes(labels, delay_s):
    out = labels.copy()
    k = int(round(delay_s * FPS))
    for i in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
        a, b = (labels[i - 1], labels[i])
        if a in ('H', 'A') and b in ('H', 'A') and (i + k < len(labels)) and (labels[i:i + k + 1] == b).all():
            out[i:i + k] = a
    return out
