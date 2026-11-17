import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.io import load_atd
from tools.false_restart import false_restarts
from tools.dead_support import support
V24 = os.path.join(C.ROOT, 'experiments', 'pipeline_approach_v24', 'ours_pipeline_approach_v24.csv')
OUT_DIR = os.path.join(C.ROOT, 'experiments', 'pipeline_approach_v26')
MAX_S = 5.0
FPS = 25

def short_dead_support(lab, ball):
    inside = np.zeros(len(lab), bool)
    for a, b in C.HALVES.values():
        inside[a:b + 1] = True
    out = lab.copy()
    st = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1]
    en = np.r_[st[1:], len(out)]
    v = out[st]
    n_flip = 0
    for i in range(1, len(st) - 1):
        if v[i] != 'D' or en[i] - st[i] >= MAX_S * FPS or (not inside[st[i]:en[i]].all()):
            continue
        if v[i - 1] not in ('H', 'A') or v[i + 1] not in ('H', 'A'):
            continue
        if support(ball, st[i], en[i]) is not None:
            continue
        team = v[i - 1] if en[i - 1] - st[i - 1] >= en[i + 1] - st[i + 1] else v[i + 1]
        out[st[i]:en[i]] = team
        n_flip += 1
    return (out, n_flip)

def main():
    atd = load_atd()
    lab = pd.read_csv(V24, keep_default_na=False).label.values.astype('<U1')
    lab, n_fr, _ = false_restarts(lab, atd['ball'])
    lab, n_ds = short_dead_support(lab, atd['ball'])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'ours_pipeline_approach_v26.csv')
    pd.DataFrame({'frame': np.arange(len(lab)), 'label': lab}).to_csv(out, index=False)
    import hashlib
    print('saques falsos: {}; paradas cortas sin motivo: {}'.format(n_fr, n_ds))
    print('-> {} sha256 {}'.format(out, hashlib.sha256(open(out, 'rb').read()).hexdigest()))
if __name__ == '__main__':
    main()
