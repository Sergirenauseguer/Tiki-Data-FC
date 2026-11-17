import argparse
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.io import load_atd
FPS = 25

def ball_speed(ball):
    xy = ball[:, :2]
    v = np.full_like(xy, np.nan)
    v[1:-1] = (xy[2:] - xy[:-2]) * (FPS / 2.0)
    return np.hypot(v[:, 0], v[:, 1])

def apply(lab, atd, v_slow=3.0, d_opp=12.0, min_s=1.0, look_s=3.0, grace_s=0.5):
    ball, players, team = (atd['ball'], atd['players'], atd['team'])
    speed = ball_speed(ball)
    k = 13
    pad = np.pad(speed, k // 2, mode='edge')
    sm = np.array([np.nanmedian(pad[i:i + k]) for i in range(len(speed))]) if False else None
    out = lab.copy()
    st = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1]
    grace = int(grace_s * FPS)
    n_fix, n_fr = (0, 0)
    for i in range(1, len(st)):
        t = st[i]
        cur = out[t]
        if out[t - 1] != 'D' or cur not in ('H', 'A'):
            continue
        opp = 1 if cur == 'H' else 0
        hi = min(len(out), t + int(30 * FPS))
        seg_speed = speed[t:hi]
        k2 = min(k, len(seg_speed))
        padded = np.pad(seg_speed, (k2 // 2,), mode='edge')
        smooth = np.array([np.nanmedian(padded[j:j + k2]) for j in range(len(seg_speed))])
        end, bad = (t, 0)
        for f in range(t, hi):
            ok = True
            if np.isnan(ball[f, 0]) or not smooth[f - t] < v_slow:
                ok = False
            else:
                d = np.hypot(players[f, :, 0] - ball[f, 0], players[f, :, 1] - ball[f, 1])
                d_o = np.nanmin(np.where(team == opp, d, np.nan))
                if not (np.isfinite(d_o) and d_o >= d_opp):
                    ok = False
            if ok:
                end, bad = (f + 1, 0)
            else:
                bad += 1
                if bad > grace:
                    break
        if end - t >= min_s * FPS:
            out[t:end] = 'D'
            n_fix += 1
            n_fr += end - t
    return (out, n_fix, n_fr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--v-slow', type=float, default=3.0)
    ap.add_argument('--d-opp', type=float, default=12.0)
    a = ap.parse_args()
    lab = pd.read_csv(a.csv, keep_default_na=False).label.values.astype('<U1')
    out_lab, n_fix, n_fr = apply(lab, load_atd(), a.v_slow, a.d_opp)
    out = a.csv.replace('.csv', '_restartwait.csv')
    pd.DataFrame({'frame': np.arange(len(out_lab)), 'label': out_lab}).to_csv(out, index=False)
    print('reanudaciones retrasadas al golpeo: {} ({} frames a D) -> {}'.format(n_fix, n_fr, out))
if __name__ == '__main__':
    main()
