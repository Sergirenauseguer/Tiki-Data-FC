import argparse
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.io import load_atd
FPS = 25

def support(ball, a, b, line_m=1.0, miss_frac=0.2, long_s=30.0):
    seg = ball[a:b, :2]
    miss = np.isnan(seg[:, 0])
    if miss.mean() >= miss_frac:
        return 'hueco'
    if b - a >= long_s * FPS:
        return 'larga'
    x, y = (seg[~miss, 0], seg[~miss, 1])
    if len(x):
        out = (x < 0) | (x > C.PITCH_LENGTH) | (y < 0) | (y > C.PITCH_WIDTH)
        if out.any():
            return 'fuera'
        near = (x < line_m) | (x > C.PITCH_LENGTH - line_m) | (y < line_m) | (y > C.PITCH_WIDTH - line_m)
        if near.any():
            return 'línea'
    return None

def apply(lab, ball):
    inside = np.zeros(len(lab), bool)
    for a, b in C.HALVES.values():
        inside[a:b + 1] = True
    out = lab.copy()
    st = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1]
    en = np.r_[st[1:], len(out)]
    v = out[st]
    n_flip, n_fr = (0, 0)
    for i in range(1, len(st) - 1):
        if v[i] != 'D' or not inside[st[i]:en[i]].all():
            continue
        if v[i - 1] not in ('H', 'A') or v[i + 1] not in ('H', 'A'):
            continue
        if support(ball, st[i], en[i]) is not None:
            continue
        left_len = en[i - 1] - st[i - 1]
        right_len = en[i + 1] - st[i + 1]
        team = v[i - 1] if left_len >= right_len else v[i + 1]
        out[st[i]:en[i]] = team
        n_flip += 1
        n_fr += en[i] - st[i]
    return (out, n_flip, n_fr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    a = ap.parse_args()
    lab = pd.read_csv(a.csv, keep_default_na=False).label.values.astype('<U1')
    out_lab, n_flip, n_fr = apply(lab, load_atd()['ball'])
    out = a.csv.replace('.csv', '_deadsupport.csv')
    pd.DataFrame({'frame': np.arange(len(out_lab)), 'label': out_lab}).to_csv(out, index=False)
    print('tramos D sin motivo vueltos a vivo: {} ({} frames) -> {}'.format(n_flip, n_fr, out))
if __name__ == '__main__':
    main()
