import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import chain_generic as CG
from src import external_idsse as EI
from src import pipeline_approach_v14_physics as V14
from src import pipeline_approach_v17_learned as V17
from src import pipeline_approach_v19_clean as V19
from src import spell_decoding as SD
from src import chain_features as CF
NAME = 'pipeline_approach_v20'
OUT_DIR = os.path.join(C.ROOT, 'experiments', NAME)
RULES_CSV = os.path.join(OUT_DIR, 'rules_v15.csv')
DECODER_JSON = os.path.join(OUT_DIR, 'params_decoder.json')
SEEDS = V17.SEEDS
CHAIN = {'horizon_s': 0.4, 'margin_m': 1.75, 'window': 75, 'delay_s': 0.8}
PASS = 'p2'

def apply_chain_constants():
    CG.HORIZON_S, CG.MARGIN_M, CG.WINDOW = (CHAIN['horizon_s'], CHAIN['margin_m'], CHAIN['window'])
    V14.TEAM_CHANGE_DELAY_S = CHAIN['delay_s']
    CF.CHAIN_TAG = '_' + PASS

def tag(deg, train):
    return '{}_deg{}_{}'.format(PASS, deg, train)

def step_rules():
    from src.io import load_atd
    apply_chain_constants()
    t = time.time()
    lab = V19.rules_barca(load_atd(), V19.V13.load_tagging())
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame({'frame': np.arange(len(lab)), 'label': lab}).to_csv(RULES_CSV, index=False)
    print('1) reglas en el Barça → {} ({:.0f} s)'.format(RULES_CSV, time.time() - t), flush=True)
    return lab

def setup(deg=2):
    apply_chain_constants()
    EI.set_degradation(deg, RULES_CSV if deg == 2 else V19.RULES_CSV, suffix=PASS + '_')

def training_labels(data, train):
    if train == 'DFL':
        return data
    refs = V17.references('controlada', sorted({m for m, _ in data}))
    for (m, s), X in data.items():
        X['label'] = refs[m]
    return data

def oof_path(deg, train):
    return os.path.join(OUT_DIR, 'oof_{}.npz'.format(tag(deg, train)))

def lomo(data, cols):
    mids = sorted({m for m, _ in data})
    out = {}
    for mid in mids:
        t = time.time()
        models = V17.fit_models([X for (m, s), X in data.items() if m != mid], cols)
        X = data[mid, SEEDS[0]]
        p_dead, p_home = V17.predict(models, X, cols)
        out[mid] = (X['ch_v15_d'].values > 0.5, p_home, p_dead)
        print('  fuera {} ({:.0f} s)'.format(mid, time.time() - t), flush=True)
    return out

def compute_oof(deg, train, need_pdead=False):
    path = oof_path(deg, train)
    if os.path.isfile(path):
        z = np.load(path)
        if not need_pdead or any((k.endswith('_pdead') for k in z.files)):
            mids = sorted({k.rsplit('_', 1)[0] for k in z.files})
            return {m: (z[m + '_dead'], z[m + '_ph'], z[m + '_pdead'] if m + '_pdead' in z.files else None) for m in mids}
    setup(deg)
    data = training_labels(V17.load_external(seeds=SEEDS, chain=True), train)
    cols = V17.feature_cols(next(iter(data.values())))
    oof = lomo(data, cols)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(path, **{k: v for m, (d, ph, pdd) in oof.items() for k, v in ((m + '_dead', d), (m + '_ph', ph), (m + '_pdead', pdd))})
    return oof

def final_model(deg, train):
    path = os.path.join(OUT_DIR, 'probs_barca_{}.npz'.format(tag(deg, train)))
    if os.path.isfile(path):
        z = np.load(path)
        return (z['p_home'], z['p_dead'])
    import gc
    setup(deg)
    data = training_labels(V17.load_external(seeds=SEEDS, chain=True), train)
    cols = V17.feature_cols(next(iter(data.values())))
    train_rows = [X.iloc[::V17.STEP].copy() for X in data.values()]
    del data
    gc.collect()
    models = V17.fit_models(train_rows, cols, step=1)
    del train_rows
    gc.collect()
    XB = pd.concat([V17.barca_features(), CF.barca_chain()], axis=1)
    p_dead, p_home = V17.predict(models, XB, cols)
    np.savez_compressed(path, p_home=p_home, p_dead=p_dead)
    return (p_home, p_dead)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deg', type=int, default=2, choices=(1, 2))
    ap.add_argument('--train', default='DFL', choices=('DFL', 'controlada'))
    ap.add_argument('--stage', default='all', choices=('oof', 'all'))
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    rules = step_rules()
    print('2-4) externos degradados (v{}), señales y P(H) fuera de muestra, etiqueta {}'.format(a.deg, a.train), flush=True)
    compute_oof(a.deg, a.train, need_pdead=True)
    if a.stage == 'oof':
        return
    with open(DECODER_JSON) as fh:
        P = json.load(fh)[tag(a.deg, a.train)]
    print('5) decodificador elegido en los externos:', P, flush=True)
    p_home, p_dead = final_model(a.deg, a.train)
    from src.io import load_atd
    lab = SD.decode_rule_dead(V17.decode, rules == 'D', p_home, P['c_team'], P['c_dead'], P.get('p_rule', V19.P_DEAD_RULE), P.get('smooth_s', 0.0), P.get('min_s', 0.0), p_dead_learned=p_dead, w_rule=P.get('w_rule', 1.0))
    lab = V17.apply_definitions(lab, load_atd())
    out_csv = os.path.join(OUT_DIR, 'ours_{}.csv'.format(NAME))
    pd.DataFrame({'frame': np.arange(len(lab)), 'label': lab}).to_csv(out_csv, index=False)
    with open(os.path.join(OUT_DIR, 'params.json'), 'w') as fh:
        json.dump({'deg': a.deg, 'train': a.train, 'decoder': P, 'seeds': list(SEEDS), 'v6': {'horizon_s': CG.HORIZON_S, 'margin_m': CG.MARGIN_M, 'window': CG.WINDOW}, 'team_change_delay_s': V14.TEAM_CHANGE_DELAY_S}, fh, indent=1)
    print('-> {} ({} filas)'.format(out_csv, len(lab)))
if __name__ == '__main__':
    main()
