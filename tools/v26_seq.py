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
TAG = os.environ.get('SEQ_TAG', '')
SEQ_DIR = os.path.join(ROOT, 'experiments', 'pipeline_approach_v21_seq' + TAG)
V21 = os.path.join(ROOT, 'experiments', 'pipeline_approach_v21')
P.OUT_DIR = SEQ_DIR
import pipeline_approach_v25 as V
import noche_v21 as N
NAME = 'pipeline_approach_v26_seq' + TAG + os.environ.get('SEQ_NAME_SUF', '')
OUT = os.path.join(ROOT, 'experiments', NAME)

def setup():
    os.makedirs(OUT, exist_ok=True)
    V.OUT_DIR = OUT
    P.OUT_DIR = OUT
    P.PASOS_CSV = os.path.join(OUT, 'pasos.csv')
    P.ESTADO = os.path.join(OUT, 'estado.json')
    V.TMP = os.path.join(ROOT, 'cache', 'v26seq' + TAG)
    os.makedirs(V.TMP, exist_ok=True)
    e = json.load(open(os.path.join(SEQ_DIR, 'estado_N1.5.json')))
    v25 = json.load(open(os.path.join(ROOT, 'experiments', 'pipeline_approach_v25', 'estado.json')))
    import pandas as pd
    rc = pd.read_csv(os.path.join(SEQ_DIR, 'ref_cmp_{}.csv'.format(P.REF_KIND)))
    nota = {lam: P.nota(dict(zip(d.partido, d['spell_vs_' + P.REF_KIND])))[0] for lam, d in rc.groupby('lam')}
    e['spell_select']['lam'] = float(os.environ.get('SEQ_LAM') or max(nota, key=nota.get))
    print('λ del selector:', e['spell_select']['lam'], flush=True)
    e['post'], e['post_alt'] = (v25['post'], v25['post_alt'])
    e['seq'] = 'P(H) de tools/seq_model.py full, media de: ' + os.environ.get('SEQ_FILES', 'seq_full.npz')
    json.dump(e, open(P.ESTADO, 'w'), indent=1)
    for f in V.REUSE:
        shutil.copy(os.path.join(SEQ_DIR, f), OUT)
    from seq_state import up
    from seq_state import load_seq
    seq = load_seq()
    orig = N.barca_inputs

    def barca_inputs_seq():
        if 'barca' in N.CACHE:
            return N.CACHE['barca']
        probs, sig, pch, kick = orig()
        ph0 = np.asarray(probs[1], np.float32)
        ph = np.repeat(seq[BK + '_ph'], 5).astype(np.float32)
        ph = np.r_[ph, np.full(max(0, len(ph0) - len(ph)), 0.5, np.float32)][:len(ph0)]
        if os.environ.get('SEQ_PCH') == '1':
            pch = np.mean([up(seq_k, len(ph0)) for seq_k in [seq[BK + '_pc']]], axis=0).astype(np.float32)
        N.CACHE['barca'] = ((probs[0], ph) + tuple(probs[2:]), sig, pch, kick)
        return N.CACHE['barca']
    N.barca_inputs = barca_inputs_seq

def main():
    setup()
    e = P.estado()
    fill = [list(x) for x in e['post']]
    alt = [list(x) for x in e['post_alt']]
    V.final()
    for s in ('', '_reglas'):
        src = os.path.join(OUT, 'ours_pipeline_approach_v25{}.csv'.format(s))
        dst = os.path.join(OUT, 'ours_{}{}.csv'.format(NAME, s))
        os.replace(src, dst)
        import hashlib
        print('->', dst, hashlib.sha256(open(dst, 'rb').read()).hexdigest()[:16], flush=True)
if __name__ == '__main__':
    main()
