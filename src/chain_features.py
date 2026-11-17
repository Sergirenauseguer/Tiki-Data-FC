import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.chain_generic import run_chain
FEAT_DIR = os.path.join(C.CACHE_DIR, 'features')
CHAIN_TAG = ''

def _run_geometry(lab):
    n = len(lab)
    st = np.r_[0, np.flatnonzero(lab[1:] != lab[:-1]) + 1]
    en = np.r_[st[1:], n]
    rid = np.repeat(np.arange(len(st)), en - st)
    i = np.arange(n)
    return ((i - st[rid]) / C.FPS, (en[rid] - i) / C.FPS, (en - st)[rid] / C.FPS)

def chain_features(atd, halves, pitch):
    st = run_chain(atd, halves, pitch, upto='all')
    f = {}
    for k in ('v6', 'v13', 'v15'):
        lab = st[k]
        f['ch_%s_ha' % k] = np.where(lab == 'H', 1.0, np.where(lab == 'A', -1.0, 0.0))
        f['ch_%s_d' % k] = (lab == 'D').astype(float)
    a, b, L = _run_geometry(st['v15'])
    f['ch_v15_since'], f['ch_v15_until'], f['ch_v15_runlen'] = (np.minimum(a, 120), np.minimum(b, 120), np.minimum(L, 300))
    return pd.DataFrame(f).astype(np.float32)

def ext_chain(mid, seed):
    from src.external_idsse import deg_tag, load_degraded
    path = os.path.join(FEAT_DIR, '{}_{}{}_chain.parquet'.format(mid, deg_tag(), seed))
    if not os.path.isfile(path):
        m = load_degraded(mid, seed)
        n, h2 = (len(m['label']), int(m['half_start'][1]))
        chain_features(m, {1: (0, h2 - 1), 2: (h2, n - 1)}, (105.0, 68.0)).to_parquet(path)
    return pd.read_parquet(path)

def barca_chain():
    from src.io import load_atd
    path = os.path.join(FEAT_DIR, 'barca_chain{}.parquet'.format(CHAIN_TAG))
    if not os.path.isfile(path):
        chain_features(load_atd(), C.HALVES, (C.PITCH_LENGTH, C.PITCH_WIDTH)).to_parquet(path)
    return pd.read_parquet(path)
if __name__ == '__main__':
    from src.external_idsse import list_matches
    os.makedirs(FEAT_DIR, exist_ok=True)
    barca_chain()
    print('barca ok', flush=True)
    for mid in list_matches():
        for s in (0, 1):
            while not os.path.isfile(os.path.join(C.CACHE_DIR, 'idsse', '{}_deg{}.npz'.format(mid, s))):
                import time
                time.sleep(20)
            ext_chain(mid, s)
            print(mid, s, 'ok', flush=True)
