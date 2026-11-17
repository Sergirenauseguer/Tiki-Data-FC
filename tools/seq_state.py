import json
import os
import shutil
import sys
import numpy as np
BK = __import__('os').environ.get('SEQ_BARCA_KEY', 'barca') + '_0'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)
import pipeline_approach_v21 as P
import noche_v21 as N
V21 = os.path.join(ROOT, 'experiments', 'pipeline_approach_v21')
COPY = ('estado_N1.5.json', 'paso0.json', 'params.json', 'pd_players10.npz', 'pchange_C_v5.npz')
SEQ_FILES = os.environ.get('SEQ_FILES', 'seq_full.npz').split(',')
TAG = os.environ.get('SEQ_TAG', '')
PCH = os.environ.get('SEQ_PCH') == '1'
PD_W = float(os.environ.get('SEQ_PD_W', 1.0))
PD_FILE = os.environ.get('SEQ_PD')

def load_seq():
    zs = [np.load(os.path.join(V21, f)) for f in SEQ_FILES]
    keys = sorted({k for z in zs for k in z.files if k != 'step'})
    return {k: np.mean([z[k] for z in zs if k in z.files], axis=0) for k in keys}

def up(x5, n, step=5):
    out = np.repeat(np.asarray(x5, np.float32), step)
    if len(out) < n:
        out = np.r_[out, np.full(n - len(out), out[-1] if len(out) else 0.5, np.float32)]
    return out[:n]

def main():
    w = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    seq = load_seq()
    out_dir = os.path.join(ROOT, 'experiments', 'pipeline_approach_v21_seq' + TAG + ('' if w == 1.0 else '_w{:g}'.format(w)))
    os.makedirs(out_dir, exist_ok=True)
    for f in COPY:
        shutil.copy(os.path.join(V21, f), out_dir)
    shutil.copy(os.path.join(V21, 'estado_N1.5.json'), os.path.join(out_dir, 'estado.json'))
    z = dict(np.load(os.path.join(V21, 'oof_C_v5.npz')))
    s1 = dict(np.load(os.path.join(V21, 'noche_seed1.npz')))
    for m in P.matches():
        for c, d in ((0, z), (1, s1)):
            ph0 = d[m + '_ph']
            d[m + '_ph'] = (w * up(seq['{}_{}_ph'.format(m, c)], len(ph0)) + (1 - w) * ph0).astype(np.float32)
    if PCH:
        pc0 = dict(np.load(os.path.join(V21, 'pchange_C_v5.npz')))
        for m in P.matches():
            pc0[m] = up(seq['{}_0_pc'.format(m)], len(pc0[m]))
            s1[m + '_pch'] = up(seq['{}_1_pc'.format(m)], len(s1[m + '_pch']))
        np.savez_compressed(os.path.join(out_dir, 'pchange_C_v5.npz'), **pc0)
    if PD_FILE:
        _z = [np.load(os.path.join(V21, f)) for f in PD_FILE.split(',')]
        sd_ = {k: np.mean([z[k] for z in _z], axis=0) for k in _z[0].files if k != 'step'}
        pp = dict(np.load(os.path.join(V21, 'pd_players10.npz')))
        for m in P.matches():
            pp[m] = PD_W * up(sd_['{}_0_pd'.format(m)], len(pp[m])) + (1 - PD_W) * pp[m]
            s1[m + '_pdp'] = PD_W * up(sd_['{}_1_pd'.format(m)], len(s1[m + '_pdp'])) + (1 - PD_W) * s1[m + '_pdp']
        np.savez_compressed(os.path.join(out_dir, 'pd_players10.npz'), **pp)
    np.savez_compressed(os.path.join(out_dir, 'oof_C_v5.npz'), **z)
    np.savez_compressed(os.path.join(out_dir, 'noche_seed1.npz'), **s1)
    P.OUT_DIR = out_dir
    P.PASOS_CSV = os.path.join(out_dir, 'pasos.csv')
    orig = N.barca_inputs
    from src import config as C

    def barca_inputs_seq():
        if 'barca' in N.CACHE:
            return N.CACHE['barca']
        probs, sig, pch, kick = orig()
        ph0 = np.asarray(probs[1], np.float32)
        ph = (w * up(seq[BK + '_ph'], len(ph0)) + (1 - w) * ph0).astype(np.float32)
        if PCH:
            pch = up(seq[BK + '_pc'], len(ph0))
        if PD_FILE:
            pdb = np.mean([np.load(os.path.join(V21, f))[BK + '_pd'] for f in PD_FILE.split(',')], axis=0)
            pdp = PD_W * up(pdb, len(ph0)) + (1 - PD_W) * np.asarray(probs[4], np.float32)
            probs = probs[:4] + (pdp,)
        N.CACHE['barca'] = ((probs[0], ph) + tuple(probs[2:]), sig, pch, kick)
        return N.CACHE['barca']
    N.barca_inputs = barca_inputs_seq
    N.step_ref(add=os.environ.get('SEQ_ADD') or None)
    print('estado ->', out_dir, flush=True)
if __name__ == '__main__':
    main()
