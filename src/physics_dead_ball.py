import warnings
import numpy as np
import pandas as pd
from src.config import FPS, HALVES, PITCH_LENGTH, PITCH_WIDTH
from src.postprocess import runs_of
FAST_P90 = 3.0
REACTION_S = 1.0
MAX_BACKDATE_S = 6.0
LINE_MARGIN_M = 2.0
OUTWARD_SPEED = 2.0
MAX_FRAGMENT_S = 8.0
LIVE_P90 = 3.5

def player_p90(atd, half_window_s=0.4):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        p90 = np.nanpercentile(atd['players'][:, :, 2], 90, axis=1)
    k = int(round(half_window_s * FPS))
    return pd.Series(p90).rolling(2 * k + 1, center=True, min_periods=1).mean().values

def out_of_play_frames(kin):
    ball, bv = (kin['ball'], kin['ball_v'])
    x, y = (ball[:, 0], ball[:, 1])
    with np.errstate(invalid='ignore'):
        oob = (x < 0) | (x > PITCH_LENGTH) | (y < 0) | (y > PITCH_WIDTH)
    vis = np.isfinite(x)
    lost = np.zeros(len(x), dtype=bool)
    idx = np.flatnonzero(vis[:-1] & ~vis[1:])
    for i in idx:
        vx, vy = bv[i] if np.isfinite(bv[i]).all() else bv[i - 2] if i >= 2 else (np.nan, np.nan)
        if not (np.isfinite(vx) and np.isfinite(vy)):
            continue
        out = x[i] < LINE_MARGIN_M and vx < -OUTWARD_SPEED or (x[i] > PITCH_LENGTH - LINE_MARGIN_M and vx > OUTWARD_SPEED) or (y[i] < LINE_MARGIN_M and vy < -OUTWARD_SPEED) or (y[i] > PITCH_WIDTH - LINE_MARGIN_M and vy > OUTWARD_SPEED)
        if out:
            lost[i + 1] = True
    return (oob, lost)

def backdate_dead_starts(labels, p90s, oob, lost, look_back_s=MAX_BACKDATE_S, use_whistle=True, use_line=True, reaction_s=REACTION_S, fast_p90=FAST_P90, restart_frames=None):
    out = labels.copy()
    lb = int(round(look_back_s * FPS))
    react = int(round(reaction_s * FPS))
    n_line = n_whistle = 0
    for a, b in HALVES.values():
        for i, (s, e, lab) in enumerate(runs_of(labels, a, b + 1)):
            if lab != 'D' or s == a or labels[s - 1] not in ('H', 'A'):
                continue
            lo = max(a, s - lb)
            if restart_frames is not None and len(restart_frames):
                r = restart_frames[(restart_frames >= lo) & (restart_frames < s)]
                if len(r):
                    lo = int(r[-1]) + 1
            new = None
            if use_line:
                cand = np.flatnonzero(oob[lo:s + FPS] | lost[lo:s + FPS])
                if len(cand):
                    f = lo + cand[-1]
                    while f - 1 >= lo and oob[f - 1]:
                        f -= 1
                    if f < s:
                        new, kind = (f, 'line')
            if new is None and use_whistle:
                w = p90s[lo:s + FPS]
                fast = np.flatnonzero(w >= fast_p90)
                if len(fast):
                    f = lo + fast[-1] + 1 - react
                    if lo <= f < s:
                        new, kind = (f, 'whistle')
            if new is not None:
                out[new:s] = 'D'
                if kind == 'line':
                    n_line += 1
                else:
                    n_whistle += 1
    return (out, n_line, n_whistle)

def fill_dead_fragments(labels, p90s, restart_frames, max_s=MAX_FRAGMENT_S, live_p90=LIVE_P90):
    out = labels.copy()
    rf = np.asarray(sorted(restart_frames))
    n = 0
    for a, b in HALVES.values():
        rr = runs_of(labels, a, b + 1)
        dead = [i for i, r in enumerate(rr) if r[2] == 'D']
        for i, j in zip(dead[:-1], dead[1:]):
            if j == i + 1:
                continue
            s, e = (rr[i][1], rr[j][0])
            if e - s > max_s * FPS:
                continue
            restarted = len(rf) and np.any(np.abs(rf - s) <= FPS)
            live = np.nanmax(p90s[s:e], initial=0) > live_p90
            if restarted and live:
                continue
            out[s:e] = 'D'
            n += 1
    return (out, n)

def revive_running(labels, p90s, ball_visible, min_p90=FAST_P90, max_visible=0.5, min_s=2.0):
    out = labels.copy()
    n = 0
    for a, b in HALVES.values():
        rr = runs_of(labels, a, b + 1)
        for i, (s, e, lab) in enumerate(rr):
            if lab != 'D' or e - s < min_s * FPS or i == 0:
                continue
            prev = rr[i - 1][2]
            if prev not in ('H', 'A'):
                continue
            if ball_visible[s:e].mean() >= max_visible:
                continue
            w = p90s[s:e]
            if np.isfinite(w).any() and np.nanmean(w) > min_p90:
                out[s:e] = prev
                n += 1
    return (out, n)
CATCH_RADIUS_M = 2.0
CATCH_APPROACH_M = 6.0
CATCH_MIN_GAP_S = 1.0
MAX_HOLD_S = 15.0
CHANGE_LOOKBACK_S = 6.0

def keeper_catches(kin, gk_slots):
    from src.features import half_of_frames
    ball, bv, P = (kin['ball'], kin['ball_v'], kin['players'])
    vis = np.isfinite(ball[:, 0])
    half = half_of_frames(len(ball))
    edges = np.diff(np.r_[0, (~vis).astype(np.int8), 0])
    out = []
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if s == 0 or e - s < CATCH_MIN_GAP_S * FPS or half[s] == 0:
            continue
        i = s - 1
        for t in (0, 1):
            g = P[i, gk_slots[int(half[s]), t]]
            if not np.isfinite(g).all():
                continue
            rel = g - ball[i]
            d = np.hypot(*rel)
            v = bv[i] if np.isfinite(bv[i]).all() else bv[i - 2]
            closing = np.isfinite(v).all() and np.dot(v, rel) > 0
            if d < CATCH_RADIUS_M or (d < CATCH_APPROACH_M and closing):
                out.append((int(s), t, int(e)))
                break
    return out

def keeper_catch_alive(labels, catches, max_hold_s=MAX_HOLD_S, lookback_s=CHANGE_LOOKBACK_S):
    out = labels.copy()
    code = np.array(['H', 'A'])
    hold = int(round(max_hold_s * FPS))
    lb = int(round(lookback_s * FPS))
    n = 0
    for c, t, gap_end in catches:
        own, other = (code[t], code[1 - t])
        end = min(gap_end, c + hold)
        if gap_end - c > hold:
            continue
        seg = out[c:end]
        if not np.isin(seg, ['D', other]).any():
            pre = out[max(0, c - lb):c]
            if not ((pre == own).any() and (pre == other).any()):
                continue
        out[c:end] = own
        k = c
        while k > c - lb and out[k - 1] == own:
            k -= 1
        if k > c - lb and out[k - 1] == other:
            out[k:c] = other
        n += 1
    return (out, n)
