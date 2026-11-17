import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src.frame_features import build_features, swap_teams
sys.path.insert(0, os.path.join(C.CHALLENGE_ROOT, 'evaluation'))
import possession_metrics as pm
NAME = 'pipeline_approach_v17_learned'
OUT_DIR = os.path.join(C.ROOT, 'experiments', NAME)
FEAT_DIR = os.path.join(C.CACHE_DIR, 'features')
SEEDS = (0, 1)
EXT_PITCH = (105.0, 68.0)
STEP = 5
LGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1, n_jobs=8, random_state=0, deterministic=True, force_row_wise=True)

def ext_halves(m):
    n, h2 = (len(m['label']), int(m['half_start'][1]))
    return {1: (0, h2 - 1), 2: (h2, n - 1)}

def ext_features(mid, seed):
    from src.external_idsse import deg_tag, load_degraded
    os.makedirs(FEAT_DIR, exist_ok=True)
    path = os.path.join(FEAT_DIR, '{}_{}{}.parquet'.format(mid, deg_tag(), seed))
    if os.path.isfile(path):
        return pd.read_parquet(path)
    m = load_degraded(mid, seed)
    X = build_features(m, ext_halves(m), EXT_PITCH)
    X['label'] = m['label']
    X.to_parquet(path)
    return X

def barca_features():
    from src.io import load_atd
    path = os.path.join(FEAT_DIR, 'barca_tactical.parquet')
    if os.path.isfile(path):
        return pd.read_parquet(path)
    os.makedirs(FEAT_DIR, exist_ok=True)
    X = build_features(load_atd(), C.HALVES, (C.PITCH_LENGTH, C.PITCH_WIDTH))
    X.to_parquet(path)
    return X

def load_external(seeds=SEEDS, chain=False):
    from src.external_idsse import list_matches
    data = {}
    for mid in list_matches():
        for s in seeds:
            data[mid, s] = ext_features(mid, s)
            if chain:
                from src.chain_features import ext_chain
                data[mid, s] = pd.concat([data[mid, s], ext_chain(mid, s)], axis=1)
            print('  señales', mid, 'semilla', s, flush=True)
    return data
DROP = ('label', 'n_players', 'n_out_players')

def feature_cols(X):
    return [c for c in X.columns if c not in DROP]

def fit_models(frames, cols, step=STEP):
    import lightgbm as lgb
    T = pd.concat([X.iloc[::step] for X in frames], ignore_index=True)
    dead = lgb.LGBMClassifier(**LGB_PARAMS)
    dead.fit(T[cols], (T['label'] == 'D').astype(int))
    live = T[T['label'] != 'D']
    sw = swap_teams(live[cols])
    Xt = pd.concat([live[cols], sw], ignore_index=True)
    yt = np.r_[(live['label'] == 'H').astype(int).values, (live['label'] == 'A').astype(int).values]
    team = lgb.LGBMClassifier(**LGB_PARAMS)
    team.fit(Xt, yt)
    return (dead, team)

def predict(models, X, cols):
    dead, team = models
    pd_ = dead.predict_proba(X[cols])[:, 1]
    ph = 0.5 * (team.predict_proba(X[cols])[:, 1] + 1.0 - team.predict_proba(swap_teams(X[cols]))[:, 1])
    return (pd_.astype(np.float32), ph.astype(np.float32))
_RUN = None

def _viterbi_core(lp, trans):
    global _RUN
    if _RUN is None:
        _RUN = _compile()
    return _RUN(lp, trans)

def _compile():
    import numba

    @numba.njit
    def run(lp, trans):
        n, k = lp.shape
        back = np.zeros((n, k), np.int8)
        score = lp[0].copy()
        new = np.empty(k)
        for t in range(1, n):
            for j in range(k):
                best, arg = (-1e+18, 0)
                for i in range(k):
                    v = score[i] - trans[i, j]
                    if v > best:
                        best, arg = (v, i)
                new[j] = best + lp[t, j]
                back[t, j] = arg
            for j in range(k):
                score[j] = new[j]
        path = np.empty(n, np.int8)
        path[n - 1] = np.argmax(score)
        for t in range(n - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path
    return run

def decode(p_dead, p_home, c_team, c_dead, eps=0.0001):
    pdd = np.clip(p_dead, eps, 1 - eps)
    ph = np.clip(p_home, eps, 1 - eps)
    lp = np.column_stack([np.log1p(-pdd) + np.log(ph), np.log1p(-pdd) + np.log1p(-ph), np.log(pdd)])
    trans = np.array([[0, c_team, c_dead], [c_team, 0, c_dead], [c_dead, c_dead, 0]], float)
    path = _viterbi_core(lp.astype(np.float64), trans)
    return np.array(['H', 'A', 'D'])[path]

def load_params():
    with open(os.path.join(OUT_DIR, 'params.json')) as fh:
        return json.load(fh)

def score(ref, lab):
    r = pm.evaluate(pm.label_codes(np.asarray(ref)), pm.label_codes(np.asarray(lab)), bootstrap_n=0)
    return (r['spell_f1_mean'], r['macro_f1'])

def per_min(lab):
    lab = np.asarray(lab)
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.r_[0, ch]
    return np.sum(lab[starts] != 'D') / (len(lab) / C.FPS / 60)
GRID_T = (2, 5, 10, 20, 30, 45, 60, 90, 130)
GRID_D = (5, 10, 20, 40, 80)

def lomo(data, cols):
    mids = sorted({m for m, _ in data})
    cache = os.path.join(OUT_DIR, 'oof_lomo.npz')
    if os.path.isfile(cache) and set(np.load(cache).files) >= {m + '_p0' for m in mids}:
        z = np.load(cache)
        return {m: (z[m + '_p0'], z[m + '_p1'], data[m, SEEDS[0]]['label'].values.astype('<U1')) for m in mids}
    oof = {}
    for mid in mids:
        t = time.time()
        train = [X for (m, s), X in data.items() if m != mid]
        models = fit_models(train, cols)
        X = data[mid, SEEDS[0]]
        oof[mid] = predict(models, X, cols) + (X['label'].values.astype('<U1'),)
        y = oof[mid][2]
        from sklearn.metrics import roc_auc_score
        live = y != 'D'
        print('  LOMO {}: AUC dead {:.3f} · AUC team {:.3f} ({:.0f} s)'.format(mid, roc_auc_score(y == 'D', oof[mid][0]), roc_auc_score(y[live] == 'H', oof[mid][1][live]), time.time() - t), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(cache, **{m + '_p' + str(k): oof[m][k] for m in mids for k in (0, 1)})
    return oof

def references(kind, mids):
    from src.external_idsse import adapt_controlled, adapt_labels, load_events, load_match
    out = {}
    for mid in mids:
        y = load_match(mid)['label']
        if kind == 'DFL':
            out[mid] = y
        elif kind == 'adaptada':
            out[mid] = adapt_labels(y)
        else:
            out[mid] = adapt_controlled(y, load_events(mid))
    return out

def tune(oof, refs):
    rows = []
    for ct in GRID_T:
        for cd in GRID_D:
            s = [score(refs[m], decode(p0, p1, ct, cd)) for m, (p0, p1, y) in oof.items()]
            pmn = np.mean([per_min(decode(p0, p1, ct, cd)) for p0, p1, y in list(oof.values())[:2]])
            rows.append({'c_team': ct, 'c_dead': cd, 'spell': np.mean([a for a, _ in s]), 'macro': np.mean([b for _, b in s]), 'pos_min': pmn})
            print('  c_team {:4d} c_dead {:4d}: spell {:.3f} macro {:.3f} pos/min {:.2f}'.format(ct, cd, rows[-1]['spell'], rows[-1]['macro'], pmn), flush=True)
    return pd.DataFrame(rows)

def apply_definitions(lab, atd):
    from src import postprocess
    from src.pipeline_approach_v13_physics import load_tagging
    out = lab.copy()
    n = len(out)
    inside = np.zeros(n, bool)
    for a, b in C.HALVES.values():
        inside[a:b + 1] = True
    out[~inside] = 'D'
    try:
        tagging = load_tagging()
        ko = postprocess.detect_half_kickoffs(tagging, atd['ball'][:, :2])
        out = postprocess.force_pre_kickoff_dead(out, ko)
    except Exception as err:
        print('  (sin saques detectados: {})'.format(err))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', choices=('DFL', 'adaptada', 'controlada'), default='controlada')
    ap.add_argument('--chain', action='store_true', help='añade la cadena de reglas como señales')
    a = ap.parse_args()
    global NAME, OUT_DIR
    if a.chain:
        NAME = NAME + '_chain'
        OUT_DIR = os.path.join(C.ROOT, 'experiments', NAME)
    print('1) señales', flush=True)
    data = load_external(chain=a.chain)
    cols = feature_cols(next(iter(data.values())))
    print('2) dejar un partido fuera', flush=True)
    oof = lomo(data, cols)
    print('3) costes del decodificador (referencia {})'.format(a.ref), flush=True)
    grid = tune(oof, references(a.ref, list(oof)))
    best = grid.sort_values('spell', ascending=False).iloc[0]
    ct, cd = (float(best.c_team), float(best.c_dead))
    print('   elegido c_team {} c_dead {} (spell {:.3f})'.format(ct, cd, best.spell))
    print('4) modelos con los 7 partidos → Barça', flush=True)
    import gc
    train = [X.iloc[::STEP].copy() for X in data.values()]
    del data, oof
    gc.collect()
    models = fit_models(train, cols, step=1)
    del train
    gc.collect()
    XB = barca_features()
    if a.chain:
        from src.chain_features import barca_chain
        XB = pd.concat([XB, barca_chain()], axis=1)
    p_dead, p_home = predict(models, XB, cols)
    from src.io import load_atd
    lab = apply_definitions(decode(p_dead, p_home, ct, cd), load_atd())
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, 'ours_{}.csv'.format(NAME))
    pd.DataFrame({'frame': np.arange(len(lab)), 'label': lab}).to_csv(out_csv, index=False)
    np.savez_compressed(os.path.join(OUT_DIR, 'probs_barca.npz'), p_dead=p_dead, p_home=p_home)
    grid.to_csv(os.path.join(OUT_DIR, 'grid_{}.csv'.format(a.ref)), index=False)
    with open(os.path.join(OUT_DIR, 'params.json'), 'w') as fh:
        json.dump({'c_team': ct, 'c_dead': cd, 'ref': a.ref, 'seeds': SEEDS}, fh)
    from src.validation import score_public
    print('   público:', score_public(lab), ' pos/min Barça {:.2f}'.format(per_min(lab)))
    print('->', out_csv)
if __name__ == '__main__':
    main()
