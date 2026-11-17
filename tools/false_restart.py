import argparse
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.io import load_atd
FPS = 25

def ball_speed_accel(ball):
    xy = ball[:, :2]
    v = np.full_like(xy, np.nan)
    v[1:-1] = (xy[2:] - xy[:-2]) * (FPS / 2.0)
    speed = np.hypot(v[:, 0], v[:, 1])
    dv = np.full_like(xy, np.nan)
    dv[1:-1] = (v[2:] - v[:-2]) * (FPS / 2.0)
    accel = np.hypot(dv[:, 0], dv[:, 1])
    return (speed, accel)

def false_restarts(lab, ball, v_still=1.0, still_s=1.0, look_s=6.0, touch_acc=22.0):
    speed, accel = ball_speed_accel(ball)
    out = lab.copy()
    st = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1]
    n_fixed, n_frames = (0, 0)
    for i in range(1, len(st)):
        t = st[i]
        if out[t - 1] != 'D' or out[t] not in ('H', 'A'):
            continue
        hi = min(len(out), t + int(look_s * FPS))
        win_speed = speed[t:hi]
        still = win_speed < v_still
        k = int(still_s * FPS)
        run = 0
        lull_start = lull_end = -1
        for j, s in enumerate(still):
            if s:
                run += 1
                if run >= k and lull_start < 0:
                    lull_start = j - run + 1
            else:
                if lull_start >= 0:
                    lull_end = j
                    break
                run = 0
        if lull_start < 0:
            continue
        if lull_end < 0:
            lull_end = len(still)
        a0 = t + FPS // 2
        seg = accel[a0:t + lull_start]
        if len(seg) and np.nanmax(seg) >= touch_acc:
            continue
        out[t:t + lull_end] = 'D'
        n_fixed += 1
        n_frames += lull_end
    return (out, n_fixed, n_frames)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--v-still', type=float, default=1.0)
    ap.add_argument('--still-s', type=float, default=1.0)
    ap.add_argument('--look-s', type=float, default=6.0)
    ap.add_argument('--touch-acc', type=float, default=22.0)
    a = ap.parse_args()
    lab = pd.read_csv(a.csv, keep_default_na=False).label.values.astype('<U1')
    out_lab, n_fixed, n_frames = false_restarts(lab, load_atd()['ball'], a.v_still, a.still_s, a.look_s, a.touch_acc)
    out = a.csv.replace('.csv', '_falserestart.csv')
    pd.DataFrame({'frame': np.arange(len(out_lab)), 'label': out_lab}).to_csv(out, index=False)
    print('saques falsos corregidos: {} ({} frames a D) -> {}'.format(n_fixed, n_frames, out))
if __name__ == '__main__':
    main()
