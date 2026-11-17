import warnings
import numpy as np
from src.config import FPS, HALVES
CTRL_RADIUS_M = 1.5
CTRL_REL_SPEED = 3.0
CTRL_MIN_FRAMES = 3
CONTEST_RADIUS_M = 1.5

def _central_velocity(xy, half_window=2):
    v = np.full_like(xy, np.nan)
    k = half_window
    v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * FPS / (2 * k)
    return v

def kinematics(atd):
    ball = atd['ball'][:, :2].astype(np.float32)
    players = atd['players'][:, :, :2].astype(np.float32)
    return {'ball': ball, 'ball_v': _central_velocity(ball), 'players': players, 'players_v': _central_velocity(players), 'team': atd['team']}

def frame_state(kin, radius=CTRL_RADIUS_M, rel_speed=CTRL_REL_SPEED, contest_radius=CONTEST_RADIUS_M):
    ball, bv, P, PV, team = (kin['ball'], kin['ball_v'], kin['players'], kin['players_v'], kin['team'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        d = np.hypot(*(P - ball[:, None, :]).transpose(2, 0, 1))
        rel = np.hypot(*(PV - bv[:, None, :]).transpose(2, 0, 1))
    d_f = np.where(np.isfinite(d), d, np.inf)
    d_home = d_f[:, team == 0].min(axis=1)
    d_away = d_f[:, team == 1].min(axis=1)
    near = np.argmin(d_f, axis=1)
    dn = d_f[np.arange(len(d_f)), near]
    rn = rel[np.arange(len(d_f)), near]
    ctrl = (dn < radius) & (np.where(np.isfinite(rn), rn, np.inf) < rel_speed)
    slot = np.where(ctrl, near, -1)
    tm = np.where(ctrl, team[near], -1).astype(np.int8)
    contest = (d_home < contest_radius) & (d_away < contest_radius)
    return {'slot': slot, 'team': tm, 'contest': contest, 'd_home': d_home, 'd_away': d_away, 'd_near': dn, 'rel_near': rn}

def control_episodes(state, min_frames=CTRL_MIN_FRAMES, max_gap=2):
    slot = np.where(state['contest'], -1, state['slot'])
    team = np.where(state['contest'], -1, state['team'])
    out = []
    n = len(slot)
    i = 0
    while i < n:
        if slot[i] < 0:
            i += 1
            continue
        s, p, j, last = (i, slot[i], i + 1, i)
        while j < n and (slot[j] == p or (slot[j] < 0 and j - last <= max_gap)):
            if slot[j] == p:
                last = j
            j += 1
        e = last + 1
        if e - s >= min_frames:
            out.append((s, e, int(team[s]), int(p)))
        i = e
    return out

def confirmed_changes(episodes, confirm_frames):
    keep = []
    cur = None
    for i, ep in enumerate(episodes):
        s, e, t, _ = ep
        if cur is None or t == cur:
            keep.append(ep)
            cur = t
            continue
        nxt = episodes[i + 1] if i + 1 < len(episodes) else None
        if e - s >= confirm_frames or (nxt is not None and nxt[2] == t):
            keep.append(ep)
            cur = t
    return keep

def possession_from_episodes(episodes, base, dead_mask=None, confirm_frames=0):
    out = base.copy()
    dead = base == 'D' if dead_mask is None else dead_mask
    code = np.array(['H', 'A'])
    for a, b in HALVES.values():
        b += 1
        live = ~dead[a:b] & (base[a:b] != 'X')
        edges = np.diff(np.r_[0, live.astype(np.int8), 0])
        for s, e in zip(np.flatnonzero(edges == 1) + a, np.flatnonzero(edges == -1) + a):
            eps = [ep for ep in episodes if s <= ep[0] < e]
            if confirm_frames:
                eps = confirmed_changes(eps, confirm_frames)
            if not eps:
                continue
            cur, pos = (None, s)
            for es, ee, t, _ in eps:
                lab = code[t]
                if cur is None:
                    cur, pos = (lab, s)
                elif lab != cur:
                    out[pos:es] = cur
                    cur, pos = (lab, es)
            out[pos:e] = cur
    return out
STILL_SPEED = 0.7
STILL_MIN_S = 1.0
KICK_SPEED = 3.0
KICK_WITHIN_S = 1.5
GAP_MIN_S = 3.0
ONSET_SPEED = 1.0
SLOW_SPEED = 2.5
FAST_SPEED = 6.0

def restart_events(kin):
    ball = kin['ball']
    sp = np.hypot(*kin['ball_v'].T)
    vis = np.isfinite(ball[:, 0])
    n = len(sp)
    out = []
    still = vis & (sp < STILL_SPEED)
    edges = np.diff(np.r_[0, still.astype(np.int8), 0])
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if e - s < STILL_MIN_S * FPS:
            continue
        w = sp[e:min(n, e + int(KICK_WITHIN_S * FPS))]
        if np.nanmax(w, initial=0) > KICK_SPEED:
            k = e
            while k + 1 < n and np.isfinite(sp[k]) and (sp[k] < ONSET_SPEED):
                k += 1
            out.append((int(k), 'quieto'))
    slow = vis & (sp < SLOW_SPEED)
    edges = np.diff(np.r_[0, slow.astype(np.int8), 0])
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if e - s < STILL_MIN_S * FPS:
            continue
        w = sp[e:min(n, e + int(KICK_WITHIN_S * FPS))]
        if np.nanmax(w, initial=0) > FAST_SPEED:
            out.append((int(e), 'lento'))
    edges = np.diff(np.r_[0, (~vis).astype(np.int8), 0])
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if e - s < GAP_MIN_S * FPS or e >= n:
            continue
        w = sp[e:min(n, e + FPS // 2)]
        if np.nanmax(w, initial=0) > KICK_SPEED:
            out.append((int(e), 'reaparece'))
    out.sort()
    dedup = []
    for f, t in out:
        if dedup and f - dedup[-1][0] < FPS // 2:
            continue
        dedup.append((f, t))
    return dedup

def snap_dead_ends(labels, events, before_s=6.0, after_s=3.0, bias_s=1.0, min_dead_s=3.0):
    out = labels.copy()
    fr = np.array([f for f, _ in events])
    if not len(fr):
        return (out, 0)
    n_moved = 0
    from src.postprocess import runs_of
    for a, b in HALVES.values():
        rr = runs_of(labels, a, b + 1)
        for i, (s, e, lab) in enumerate(rr):
            if lab != 'D' or e - s < min_dead_s * FPS or i + 1 >= len(rr):
                continue
            nxt_lab = rr[i + 1][2]
            nxt_end = rr[i + 1][1]
            lo, hi = (e - int(before_s * FPS), e + int(after_s * FPS))
            c = fr[(fr >= max(lo, s + FPS)) & (fr <= min(hi, nxt_end - 1))]
            if not len(c):
                continue
            new_e = int(c[np.argmin(np.abs(c - (e - bias_s * FPS)))])
            if new_e == e:
                continue
            if new_e > e:
                out[e:new_e] = 'D'
            else:
                out[new_e:e] = nxt_lab
            n_moved += 1
    return (out, n_moved)

def snap_team_changes(labels, episodes, window_s=2.0, drop_unsupported=False):
    out = labels.copy()
    w = int(round(window_s * FPS))
    starts = {0: np.array([s for s, e, t, _ in episodes if t == 0]), 1: np.array([s for s, e, t, _ in episodes if t == 1])}
    code = {'H': 0, 'A': 1}
    ch = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    n_moved = n_drop = 0
    for k, i in enumerate(ch):
        a, b = (labels[i - 1], labels[i])
        if a not in code or b not in code:
            continue
        prev_b = ch[k - 1] if k else 0
        next_b = ch[k + 1] if k + 1 < len(ch) else len(labels)
        st = starts[code[b]]
        c = st[(st >= max(prev_b + 1, i - w)) & (st <= min(next_b - 1, i + w))]
        if len(c):
            j = int(c[np.argmin(np.abs(c - i))])
            if j > i:
                out[i:j] = a
            elif j < i:
                out[j:i] = b
            n_moved += j != i
        elif drop_unsupported:
            out[i:next_b] = a
            n_drop += 1
    return (out, n_moved, n_drop)
