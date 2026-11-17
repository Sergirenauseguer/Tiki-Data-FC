import argparse
import csv
import json
import os
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import chain_features as CF
from src import external_idsse as EI
from src import external_metrica as EM
from src import pipeline_approach_v17_learned as V17
from src import pipeline_approach_v20 as V20
from src import spell_decoding as SD
NAME = 'pipeline_approach_v21'
REF_KIND = os.environ.get('MTAH_REF', 'smooth')
OUT_DIR = os.path.join(C.ROOT, 'experiments', NAME + ('_raw' if REF_KIND == 'raw' else ''))
PASOS_CSV = os.path.join(OUT_DIR, 'pasos.csv')
SEEDS = (0, 1)
STEP = V17.STEP
SPLITS = {'A': ('DFL-MAT-J03WMX', 'DFL-MAT-J03WQQ', 'metrica_1'), 'B': ('DFL-MAT-J03WOY', 'DFL-MAT-J03WR9', 'metrica_3')}
DECODER = {'c_team': 10.0, 'c_dead': 80.0, 'min_s': 2.0, 'smooth_s': 0.0}
_FRAMES, _REFS = ({}, {})

def setup(deg=2):
    V20.setup(2)
    if deg in (3, 4, 5, 6):
        EI.set_degradation(deg, V20.RULES_CSV, suffix=V20.PASS + '_')
    _FRAMES.clear()
    os.makedirs(OUT_DIR, exist_ok=True)

def is_metrica(mid):
    return mid.startswith('metrica')

def matches():
    return EI.list_matches() + EM.list_matches()

def metrica_features(mid, seed):
    from src.frame_features import build_features
    path = os.path.join(V17.FEAT_DIR, '{}_{}{}_full.parquet'.format(mid, EI.deg_tag(), seed))
    if not os.path.isfile(path):
        m = EM.load_degraded(mid, seed)
        halves = V17.ext_halves(m)
        X = pd.concat([build_features(m, halves, EM.PITCH), CF.chain_features(m, halves, EM.PITCH)], axis=1)
        X['label'] = m['label']
        os.makedirs(V17.FEAT_DIR, exist_ok=True)
        X.to_parquet(path)
    return pd.read_parquet(path)
_BC_MODEL = {}

def bc_features(mid, seed):
    from src.broadcast_fill import fit_model, simulate_fill
    from src.frame_features import build_features
    path = os.path.join(V17.FEAT_DIR, '{}_{}{}_bc.parquet'.format(mid, EI.deg_tag(), seed))
    if not os.path.isfile(path):
        if 'm' not in _BC_MODEL:
            _BC_MODEL['m'] = fit_model()
        m = EM.load_degraded(mid, seed) if is_metrica(mid) else EI.load_degraded(mid, seed)
        m2 = simulate_fill(m, match_data(mid)['ball'], _BC_MODEL['m'], seed)
        halves = V17.ext_halves(m2)
        pitch = EM.PITCH if is_metrica(mid) else V17.EXT_PITCH
        X = pd.concat([build_features(m2, halves, pitch), CF.chain_features(m2, halves, pitch)], axis=1)
        X['label'] = m['label']
        X.to_parquet(path)
    return pd.read_parquet(path)

def barca_bc_features():
    from src.broadcast_fill import barca_filled_atd
    from src.frame_features import build_features
    path = os.path.join(V17.FEAT_DIR, 'barca_bc{}.parquet'.format(CF.CHAIN_TAG))
    if not os.path.isfile(path):
        atd = barca_filled_atd()
        pd.concat([build_features(atd, C.HALVES, (C.PITCH_LENGTH, C.PITCH_WIDTH)), CF.chain_features(atd, C.HALVES, (C.PITCH_LENGTH, C.PITCH_WIDTH))], axis=1).to_parquet(path)
    return pd.read_parquet(path)

def frames(mid, seed, extra=()):
    if 'bc' in extra:
        X = bc_features(mid, seed)
    elif is_metrica(mid):
        X = metrica_features(mid, seed)
    else:
        X = pd.concat([V17.ext_features(mid, seed), CF.ext_chain(mid, seed)], axis=1)
    for g in extra:
        if g != 'bc':
            X = pd.concat([X, EXTRA[g](mid, seed)], axis=1)
    return X

def match_data(mid):
    m = EM.load_match(mid) if is_metrica(mid) else EI.load_match(mid)
    return m

def retain(label, min_s, fps=25):
    out = np.asarray(label).copy()
    st = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1]
    en = np.r_[st[1:], len(out)]
    owner = 'D'
    for a, b in zip(st, en):
        v = out[a]
        if v not in ('H', 'A'):
            owner = 'D'
            continue
        if owner == 'D' or v == owner or b - a >= min_s * fps:
            owner = v
            continue
        out[a:b] = owner
    return out

def reference(mid, kind=None):
    kind = kind or REF_KIND
    key = mid if kind == REF_KIND else (mid, kind)
    if key not in _REFS:
        m = match_data(mid)
        if kind == 'c20':
            ref = retain(m['label_raw'] if is_metrica(mid) else m['label'], 2.0)
        elif is_metrica(mid):
            ref = m['label_raw'] if kind == 'raw' else m['label']
        elif kind == 'raw':
            ref = EI.adapt_labels(m['label'], 1.0)
        else:
            ref = EI.adapt_controlled(m['label'], EI.load_events(mid, m))
        _REFS[key] = (np.asarray(ref), int(m['half_start'][1]))
    return _REFS[key]

def train_label(mid, X, idsse_label):
    if is_metrica(mid) or idsse_label == 'DFL':
        return X['label'].values
    return reference(mid)[0]

def _touch_ext(mid, seed):
    from src.touch_features import touch_features
    path = os.path.join(V17.FEAT_DIR, '{}_{}{}_touch.parquet'.format(mid, EI.deg_tag(), seed))
    if not os.path.isfile(path):
        m = EM.load_degraded(mid, seed) if is_metrica(mid) else EI.load_degraded(mid, seed)
        touch_features(m, V17.ext_halves(m)).to_parquet(path)
    return pd.read_parquet(path)

def _touch_barca():
    from src.io import load_atd
    from src.touch_features import touch_features
    path = os.path.join(V17.FEAT_DIR, 'barca_touch.parquet')
    if not os.path.isfile(path):
        touch_features(load_atd(), C.HALVES).to_parquet(path)
    return pd.read_parquet(path)
TOUCH_SIGMA, TOUCH_THR = (6.0, 1.5)

def _touch_s_ext(mid, seed):
    from src.touch_features import touch_features
    path = os.path.join(V17.FEAT_DIR, '{}_{}{}_touchs6.parquet'.format(mid, EI.deg_tag(), seed))
    if not os.path.isfile(path):
        m = EM.load_degraded(mid, seed) if is_metrica(mid) else EI.load_degraded(mid, seed)
        touch_features(m, V17.ext_halves(m), sigma=TOUCH_SIGMA, thr=TOUCH_THR).to_parquet(path)
    return pd.read_parquet(path)

def _touch_s_barca():
    from src.io import load_atd
    from src.touch_features import touch_features
    path = os.path.join(V17.FEAT_DIR, 'barca_touchs6.parquet')
    if not os.path.isfile(path):
        touch_features(load_atd(), C.HALVES, thr=TOUCH_THR).to_parquet(path)
    return pd.read_parquet(path)

def _touch_v3_ext(mid, seed):
    from src.touch_features import touch_features
    path = os.path.join(V17.FEAT_DIR, '{}_{}{}_touchv3.parquet'.format(mid, EI.deg_tag(), seed))
    if not os.path.isfile(path):
        m = EM.load_degraded(mid, seed) if is_metrica(mid) else EI.load_degraded(mid, seed)
        touch_features(m, V17.ext_halves(m), thr=TOUCH_THR).to_parquet(path)
    return pd.read_parquet(path)
EXTRA = {'touch': _touch_ext, 'touch_s6': _touch_s_ext, 'touch_v3': _touch_v3_ext}

def _spells(lab):
    st, en = SD.runs(lab)
    v = np.asarray(lab)[st]
    k = (v == 'H') | (v == 'A')
    return (st[k], en[k], v[k])

def spell_f1(ref, lab):
    gs, ge, gv = _spells(ref)
    ps, pe, pv = _spells(lab)
    pairs = []
    j0 = np.searchsorted(pe, gs, side='right')
    for i in range(len(gs)):
        j = j0[i]
        while j < len(ps) and ps[j] < ge[i]:
            if pv[j] == gv[i]:
                inter = min(ge[i], pe[j]) - max(gs[i], ps[j])
                union = max(ge[i], pe[j]) - min(gs[i], ps[j])
                pairs.append((inter / union, i, j))
            j += 1
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    f1s = []
    for thr in V17.pm.SPELL_IOUS:
        ug, up, tp = (set(), set(), 0)
        for iou, i, j in pairs:
            if iou < thr:
                break
            if i in ug or j in up:
                continue
            ug.add(i)
            up.add(j)
            tp += 1
        p = tp / len(ps) if len(ps) else 0.0
        r = tp / len(gs) if len(gs) else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))

def score_match(mid, lab, kind=None):
    ref, h2 = reference(mid, kind)
    return spell_f1(np.insert(ref, h2, 'X'), np.insert(np.asarray(lab), h2, 'X'))

def nota(per_match):
    a = [v for k, v in per_match.items() if not is_metrica(k)]
    b = [v for k, v in per_match.items() if is_metrica(k)]
    if a and b:
        return (0.5 * np.mean(a) + 0.5 * np.mean(b), float(np.mean(a)), float(np.mean(b)))
    return (float(np.mean(a or b)), float(np.mean(a)) if a else np.nan, float(np.mean(b)) if b else np.nan)

def decode(probs, dec=None, fn=None):
    d = dict(DECODER, **dec or {})
    dead, ph, pdd = probs[:3]
    if fn is not None:
        return fn(probs, d)
    return SD.decode_rule_dead(V17.decode, dead, ph, d['c_team'], d['c_dead'], smooth_s=d.get('smooth_s', 0.0), min_s=d.get('min_s', 0.0))

def score_probs(oof, dec=None, fn=None):
    per = {m: score_match(m, decode(p, dec, fn)) for m, p in oof.items()}
    return (per, nota(per))

def train_rows(keys, cfg):
    out = []
    for mid, s in keys:
        k = (mid, s, tuple(cfg.get('extra', ())), cfg.get('idsse_label', 'DFL'))
        if k not in _FRAMES:
            X = frames(mid, s, cfg.get('extra', ()))
            X = X.iloc[::STEP].copy()
            X['label'] = train_label(mid, X, cfg.get('idsse_label', 'DFL')) if is_metrica(mid) or cfg.get('idsse_label', 'DFL') == 'DFL' else reference(mid)[0][::STEP]
            _FRAMES[k] = X
        out.append(_FRAMES[k])
    return out

def fit(train_mids, cfg):
    rows = train_rows([(m, s) for m in train_mids for s in cfg.get('train_seeds', SEEDS)], cfg)
    cols = V17.feature_cols(rows[0])
    t = time.time()
    k = int(cfg.get('lgb_seeds', 1))
    if k == 1:
        models = V17.fit_models(rows, cols, step=1)
    else:
        keep = dict(V17.LGB_PARAMS)
        models = []
        try:
            for r in range(k):
                V17.LGB_PARAMS.update(random_state=r)
                models.append(V17.fit_models(rows, cols, step=1))
        finally:
            V17.LGB_PARAMS.clear()
            V17.LGB_PARAMS.update(keep)
    return (models, cols, time.time() - t)

def _predict_any(models, X, cols):
    if isinstance(models, list):
        ps = [V17.predict(m, X, cols) for m in models]
        return (np.mean([p[0] for p in ps], 0).astype(np.float32), np.mean([p[1] for p in ps], 0).astype(np.float32))
    return V17.predict(models, X, cols)

def predict(models, cols, mid, cfg, seed=0):
    X = frames(mid, seed, cfg.get('extra', ()))
    p_dead, p_home = _predict_any(models, X, cols)
    return (X['ch_v15_d'].values > 0.5, p_home, p_dead)

def oof_path(tag):
    return os.path.join(OUT_DIR, 'oof_{}.npz'.format(tag))

def run_oof(cfg, tag):
    path = oof_path(tag)
    if os.path.isfile(path):
        z = np.load(path)
        return {m: (z[m + '_dead'], z[m + '_ph'], z[m + '_pd']) for m in matches() if m + '_ph' in z.files}
    mids = matches() if cfg.get('metrica_train', True) else EI.list_matches()
    out = {}
    for mid in matches():
        train = [m for m in mids if m != mid]
        models, cols, dt = fit(train, cfg)
        out[mid] = predict(models, cols, mid, cfg)
        print('  fuera {} ({:.0f} s de entrenamiento)'.format(mid, dt), flush=True)
    np.savez_compressed(path, **{m + k: v for m, p in out.items() for k, v in zip(('_dead', '_ph', '_pd'), p)})
    return out
STAGE2 = None

def accept_decoder(per_new, per_base):
    n_new, n_base = (nota(per_new)[0], nota(per_base)[0])
    worse = sum((per_new[m] < per_base[m] - 0.01 for m in per_base))
    return (n_new - n_base >= 0.003 and worse <= 3, n_new - n_base, worse)

def log_row(paso, idea, tipo, res, aceptada, nota_txt=''):
    os.makedirs(OUT_DIR, exist_ok=True)
    new = not os.path.isfile(PASOS_CSV)
    with open(PASOS_CSV, 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['paso', 'idea', 'tipo', 'nota', 'A', 'B', 'IDSSE', 'Metrica', 'aceptada', 'nota_txt'])
        w.writerow([paso, idea, tipo] + ['{:.4f}'.format(res.get(k, np.nan)) if isinstance(res.get(k), float) else '' for k in ('nota', 'A', 'B', 'IDSSE', 'Metrica')] + [aceptada, nota_txt])

def barca(cfg, dec=None, fn=None, name='final'):
    from src.io import load_atd
    pool = matches() if cfg.get('metrica_train', True) else EI.list_matches()
    models, cols, _ = fit(pool, cfg)
    XB = pd.concat([V17.barca_features(), CF.barca_chain()] + [BARCA_EXTRA[g]() for g in cfg.get('extra', ())], axis=1)
    p_dead, p_home = V17.predict(models, XB, cols)
    rules = pd.read_csv(V20.RULES_CSV, keep_default_na=False).label.values
    probs = (rules == 'D', p_home, p_dead)
    if cfg.get('stage2'):
        probs = STAGE2(pool, ['barca'], {'barca': probs}, cfg, XB=XB)['barca']
    lab = decode(probs, dec, fn)
    lab = V17.apply_definitions(lab, load_atd())
    out = os.path.join(OUT_DIR, 'barca_{}.csv'.format(name))
    pd.DataFrame({'frame': np.arange(len(lab)), 'label': lab}).to_csv(out, index=False)
    return out
BARCA_EXTRA = {'touch': _touch_barca, 'touch_s6': _touch_s_barca, 'touch_v3': _touch_s_barca}
ESTADO = os.path.join(OUT_DIR, 'estado.json')

def estado():
    return json.load(open(ESTADO)) if os.path.isfile(ESTADO) else {}

def guardar_estado(e):
    json.dump(e, open(ESTADO, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))

def final():
    e = estado()
    setup(int(e['cfg'].get('deg', 2)))
    if not os.path.isfile(V20.RULES_CSV):
        V20.step_rules()
    t = time.time()
    if e.get('spell_select'):
        import noche_v21
        tmp = noche_v21.final_select('final')
    else:
        tmp = barca_state(e, 'final')
    out = os.path.join(OUT_DIR, 'ours_{}.csv'.format(NAME))
    os.replace(tmp, out)
    json.dump({k: e[k] for k in ('cfg', 'decoder', 'alpha', 'restart_snap', 'w_rule', 'dead_src', 'spell_select') if k in e}, open(os.path.join(OUT_DIR, 'params.json'), 'w'), indent=1)
    import hashlib
    print('-> {} ({:.0f} s) sha256 {}'.format(out, time.time() - t, hashlib.sha256(open(out, 'rb').read()).hexdigest()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', type=int, default=None)
    ap.add_argument('--final', action='store_true')
    a = ap.parse_args()
    setup()
    if a.final:
        final()
        return
    {0: step0, 1: step1, 2: step2, 3: step3, 31: step3b, 32: step3c, 4: step4, 41: step4b, 5: step5, 9: stepB, 10: stepC, 11: stepD, 12: stepE}[a.step]()

def signals(mid, cfg=None):
    return frames(mid, 0)[['last_ctrl', 'since_ctrl', 'ball_vis']].reset_index(drop=True)

def change_target(ref, w=10):
    r = np.asarray(ref)
    ch = np.zeros(len(r), bool)
    idx = np.flatnonzero((r[1:] != r[:-1]) & (r[1:] != 'D') & (r[:-1] != 'D')) + 1
    for k in range(-w, w + 1):
        j = np.clip(idx + k, 0, len(r) - 1)
        ch[j] = True
    return ch

def run_pchange(cfg, tag, w=10, ctx_oof=None, suffix=''):
    import lightgbm as lgb
    path = os.path.join(OUT_DIR, 'pchange_{}{}.npz'.format(tag, suffix))
    if os.path.isfile(path):
        z = np.load(path)
        return {m: z[m] for m in z.files}
    pool = matches()
    rows = {}
    for m in pool:
        for sd in cfg.get('pch_seeds', SEEDS):
            X = frames(m, sd, cfg.get('extra', ()))
            if ctx_oof is not None:
                X = pd.concat([X.reset_index(drop=True), _ctx(ctx_oof[m][1], ctx_oof[m][2])], axis=1)
            X = X.iloc[::STEP].copy()
            X['y'] = change_target(reference(m)[0], w)[::STEP]
            rows[m, sd] = X
    cols = [c for c in V17.feature_cols(next(iter(rows.values()))) if c != 'y']
    out = {}
    for m in pool:
        T = pd.concat([X for (k, _), X in rows.items() if k != m], ignore_index=True)
        T = T[T['label'] != 'D'] if 'label' in T else T
        mdl = lgb.LGBMClassifier(**V17.LGB_PARAMS)
        mdl.fit(T[cols], T['y'].astype(int))
        Xp = frames(m, 0, cfg.get('extra', ()))
        if ctx_oof is not None:
            Xp = pd.concat([Xp.reset_index(drop=True), _ctx(ctx_oof[m][1], ctx_oof[m][2])], axis=1)
        out[m] = mdl.predict_proba(Xp[cols])[:, 1].astype(np.float32)
        print('  P(cambio) fuera', m, flush=True)
    np.savez_compressed(path, **out)
    return out

def nested_choice(cands):
    mids = list(cands[0][1])
    per = {}
    for m in mids:
        others = [x for x in mids if x != m]
        best = max(cands, key=lambda c: nota({k: c[1][k] for k in others})[0])
        per[m] = best[1][m]
    return nota(per)[0]

def kick_of(mid):
    from src import decoding_v21 as DV
    if mid == 'barca':
        from src.io import load_atd
        return DV.ball_kick(load_atd()['ball'])
    d = EM.load_degraded(mid, 0) if is_metrica(mid) else EI.load_degraded(mid, 0)
    return DV.ball_kick(d['ball'])

def full_labels(m, probs, sig, pch, e, dec=None, kick=None, pbd=None):
    from src import decoding_v21 as DV
    d = dict(e.get('decoder', DECODER), **dec or {})
    dead, ph = (probs[0], probs[1])
    c = np.full(len(ph), d['c_team'])
    if e.get('alpha'):
        c = c * (1 - e['alpha'] * pch)
    c_ha = c_ah = c
    if e.get('use_mask'):
        ch, ca = DV.control_candidates(sig, d['mask_w'])
        gap = sig['ball_vis'].values < 0.5
        c_ha = c + np.where(ca | gap, 0, d['mask_extra'])
        c_ah = c + np.where(ch | gap, 0, d['mask_extra'])
    p_dead = DV.rule_pdead(dead)
    if e.get('w_rule', 1.0) < 1.0:
        pd_learned = probs[2] if e.get('dead_src', 'paso0') == 'paso0' or len(probs) < 5 else probs[4]
        p_dead = e['w_rule'] * p_dead + (1 - e['w_rule']) * pd_learned
    c_in, c_out = (d['c_dead'], None)
    if e.get('dead_var') and pbd is not None:
        dv = e['dead_var']
        c_in = dv['c_dead'] * (1 - dv['beta'] * pbd[0])
        c_out = dv['c_dead'] * (1 - dv['beta'] * pbd[1])
    lab = SD.drop_short(DV.viterbi_tv(p_dead, ph, c_ha, c_ah, c_in, c_out=c_out), d.get('min_s', 0.0))
    if e.get('snap_w'):
        lab = DV.snap(lab, pch, e['snap_w'])
    if e.get('restart_snap'):
        k = kick if kick is not None else kick_of(m)
        lab = DV.snap_restarts_max(lab, k, e['restart_snap']['thr'], e['restart_snap']['w'])
    return lab

def _ctx(ph, pdd):
    z = np.log(np.clip(ph, 0.0001, 1 - 0.0001) / np.clip(1 - ph, 0.0001, 1))
    zd = np.log(np.clip(pdd, 0.0001, 1 - 0.0001) / np.clip(1 - pdd, 0.0001, 1))
    n = len(z)
    f = {'z0': z, 'zd0': zd}
    for name, v in (('z', z), ('zd', zd)):
        c = np.r_[0.0, np.cumsum(v)]
        i = np.arange(n)
        for s in (1, 3, 10):
            w = s * 25
            for tag, lo, hi in (('c', i - w, i + w + 1), ('p', i - w, i + 1), ('f', i, i + w + 1)):
                lo, hi = (np.clip(lo, 0, n), np.clip(hi, 0, n))
                f['{}_{}{}'.format(name, tag, s)] = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    f['z_max3'], f['z_min3'] = (maximum_filter1d(z, 151), minimum_filter1d(z, 151))
    return pd.DataFrame(f).astype(np.float32)

def _ctx_swap(F):
    G = F.copy()
    for c in F.columns:
        if c.startswith('z_') or c == 'z0' or c.startswith('z_m'):
            G[c] = -F[c].values
    G['z_max3'], G['z_min3'] = (-F['z_min3'].values, -F['z_max3'].values)
    return G

def stage2_oof(oof, target):
    import lightgbm as lgb
    params = dict(V17.LGB_PARAMS, n_estimators=200, num_leaves=31)
    ctx = {m: _ctx(p[1], p[2]) for m, p in oof.items()}
    lab = {}
    for m in oof:
        if target == 'ref':
            lab[m] = reference(m)[0]
        else:
            lab[m] = match_data(m)['label']
    out = {}
    for m in oof:
        Xs, ys = ([], [])
        for k in oof:
            if k == m:
                continue
            live = lab[k] != 'D'
            idx = np.flatnonzero(live)[::STEP]
            F = ctx[k].iloc[idx]
            Xs += [F, _ctx_swap(F)]
            y = (lab[k][idx] == 'H').astype(int)
            ys += [y, 1 - y]
        mdl = lgb.LGBMClassifier(**params)
        mdl.fit(pd.concat(Xs, ignore_index=True), np.concatenate(ys))
        F = ctx[m]
        ph2 = 0.5 * (mdl.predict_proba(F)[:, 1] + 1 - mdl.predict_proba(_ctx_swap(F))[:, 1])
        out[m] = (oof[m][0], ph2.astype(np.float32), oof[m][2])
    return out

def stage2_nested(e, target='train'):
    import lightgbm as lgb
    cfg = e['cfg']
    oof = run_oof(cfg, e['oof'])
    path = os.path.join(OUT_DIR, 'stage2_nested_{}.npz'.format(target))
    if os.path.isfile(path):
        z = np.load(path)
        return {m: (oof[m][0], z[m], oof[m][2]) for m in oof}
    params = dict(V17.LGB_PARAMS, n_estimators=200, num_leaves=31)
    ids = EI.list_matches()
    lab = {m: reference(m)[0] if target == 'ref' else match_data(m)['label'] for m in oof}

    def train_predict(probs, test):
        Xs, ys = ([], [])
        for k, p in probs.items():
            live = lab[k] != 'D'
            idx = np.flatnonzero(live)[::STEP]
            F = _ctx(p[1], p[2]).iloc[idx]
            Xs += [F, _ctx_swap(F)]
            y = (lab[k][idx] == 'H').astype(int)
            ys += [y, 1 - y]
        mdl = lgb.LGBMClassifier(**params)
        mdl.fit(pd.concat(Xs, ignore_index=True), np.concatenate(ys))
        F = _ctx(oof[test][1], oof[test][2])
        return (0.5 * (mdl.predict_proba(F)[:, 1] + 1 - mdl.predict_proba(_ctx_swap(F))[:, 1])).astype(np.float32)
    out = {}
    for i in oof:
        if is_metrica(i):
            out[i] = train_predict({k: oof[k] for k in oof if k != i}, i)
            continue
        probs = {}
        m_no_i, cols, _ = fit([m for m in ids if m != i], cfg)
        for k in EM.list_matches():
            probs[k] = predict(m_no_i, cols, k, cfg)
        for j in ids:
            if j == i:
                continue
            mdl, cols, _ = fit([m for m in ids if m not in (i, j)], cfg)
            probs[j] = predict(mdl, cols, j, cfg)
        out[i] = train_predict(probs, i)
        print('  2.ª etapa anidada', i, flush=True)
    np.savez_compressed(path, **out)
    return {m: (oof[m][0], out[m], oof[m][2]) for m in oof}
PLAYER_COLS = ['p90', 'p50', 'pmax', 'p90_c1', 'p90_c3', 'p90_c8', 'p90_p3', 'p90_f3', 'p90_p6', 'p90_f6', 'p90_p1', 'p90_f1', 'p50_c3', 'p90_dfp', 'p90_min_c5', 'n_fast', 'n_fast_c2', 'spd_h', 'spd_a', 'width_h', 'length_h', 'cx_h', 'width_a', 'length_a', 'cx_a', 'n_in_box']
LONG_BASE = ['p90', 'p50', 'n_fast', 'spd_h', 'spd_a', 'n_in_box', 'width_h', 'width_a']
LONG_WIN = (5, 10, 20)

def long_player_cols():
    return [f'{c}_{k}{w}' for c in LONG_BASE for w in LONG_WIN for k in ('c', 'p', 'f')]

def add_long(X):
    out = {}
    for c in LONG_BASE:
        x = pd.Series(np.nan_to_num(X[c].values.astype(float)))
        for w in LONG_WIN:
            n = w * 25
            out[f'{c}_c{w}'] = x.rolling(2 * n + 1, center=True, min_periods=1).mean().values
            out[f'{c}_p{w}'] = x.rolling(n, min_periods=1).mean().values
            out[f'{c}_f{w}'] = x[::-1].rolling(n, min_periods=1).mean().values[::-1]
    return pd.concat([X.reset_index(drop=True), pd.DataFrame(out)], axis=1)
RESTART_GATE = {1: 0, 2: 8}

def restart_frame(mid, seed=0, ver=1):
    from src.restart_features import restart_features
    suf = '_restart' if ver == 1 else '_restart{}'.format(ver)
    gate = RESTART_GATE[ver]
    if mid == 'barca':
        from src.io import load_atd
        path = os.path.join(V17.FEAT_DIR, 'barca{}.parquet'.format(suf))
        if not os.path.isfile(path):
            restart_features(load_atd(), C.HALVES, (C.PITCH_LENGTH, C.PITCH_WIDTH), gate).to_parquet(path)
        return pd.read_parquet(path)
    path = os.path.join(V17.FEAT_DIR, '{}_{}{}{}.parquet'.format(mid, EI.deg_tag(), seed, suf))
    if not os.path.isfile(path):
        m = EM.load_degraded(mid, seed) if is_metrica(mid) else EI.load_degraded(mid, seed)
        restart_features(m, V17.ext_halves(m), EM.PITCH if is_metrica(mid) else V17.EXT_PITCH, gate).to_parquet(path)
    return pd.read_parquet(path)

def pd_X(mid, seed=0, long=False, restart=False):
    X = V17.barca_features() if mid == 'barca' else frames(mid, seed)
    if long:
        X = add_long(X)
    if restart:
        X = pd.concat([X.reset_index(drop=True), restart_frame(mid, seed, int(restart)).reset_index(drop=True)], axis=1)
    return X

def _pd_players_model(train_mids, seeds=None, long=False, restart=False):
    import lightgbm as lgb
    rows = []
    for m in train_mids:
        for s_ in seeds or SEEDS:
            lab = train_rows([(m, s_)], {'metrica_train': True})[0]['label'].values
            X = pd_X(m, s_, long, restart).iloc[::STEP]
            rows.append(X.assign(label=lab))
    T = pd.concat(rows, ignore_index=True)
    cols = PLAYER_COLS + (long_player_cols() if long else []) + ([c for c in T.columns if c.startswith('rs_')] if restart else [])
    mdl = lgb.LGBMClassifier(**V17.LGB_PARAMS)
    mdl.fit(T[cols], (T['label'] == 'D').astype(int))
    mdl.cols_, mdl.long_, mdl.restart_ = (cols, long, restart)
    return mdl

def pd_players_predict(mdl, mid, seed=0):
    return mdl.predict_proba(pd_X(mid, seed, getattr(mdl, 'long_', False), getattr(mdl, 'restart_', False))[mdl.cols_])[:, 1].astype(np.float32)

def run_pd_players(seeds=None, long=False, restart=False):
    seeds = list(seeds) if seeds else list(SEEDS)
    name = 'pd_players10' if seeds == list(SEEDS) else 'pd_players10_s{}'.format(''.join(map(str, seeds)))
    if EI.DEGRADATION['version'] == 6:
        name += '_v6'
    rsuf = '' if not restart else '_restart' if int(restart) == 1 else '_restart{}'.format(int(restart))
    path = os.path.join(OUT_DIR, name + ('_long' if long else '') + rsuf + '.npz')
    if os.path.isfile(path):
        z = np.load(path)
        return {m: z[m] for m in z.files}
    out = {}
    for m in matches():
        mdl = _pd_players_model([k for k in matches() if k != m], seeds, long, restart)
        out[m] = pd_players_predict(mdl, m, 0)
    np.savez_compressed(path, **out)
    return out

def bound_target(ref, kind, w=10):
    d = np.asarray(ref) == 'D'
    idx = np.flatnonzero(d[1:] & ~d[:-1] if kind == 'in' else ~d[1:] & d[:-1]) + 1
    y = np.zeros(len(d), bool)
    for k in range(-w, w + 1):
        y[np.clip(idx + k, 0, len(d) - 1)] = True
    return y

def _bd_models(train_mids, w=10):
    import lightgbm as lgb
    X, Y = ([], {'in': [], 'out': []})
    for m in train_mids:
        ref = reference(m)[0]
        for sd in SEEDS:
            X.append(frames(m, sd)[PLAYER_COLS].iloc[::STEP])
            for k in Y:
                Y[k].append(bound_target(ref, k, w)[::STEP])
    T = pd.concat(X, ignore_index=True)
    out = {}
    for k in Y:
        mdl = lgb.LGBMClassifier(**V17.LGB_PARAMS)
        mdl.fit(T, np.concatenate(Y[k]).astype(int))
        out[k] = mdl
    return out

def run_pbound(w=10):
    path = os.path.join(OUT_DIR, 'pbound_{}w{}.npz'.format(EI.deg_tag(), w))
    if os.path.isfile(path):
        z = np.load(path)
        return {m: (z[m + '_in'], z[m + '_out']) for m in matches()}
    out = {}
    for m in matches():
        mods = _bd_models([k for k in matches() if k != m], w)
        X = frames(m, 0)[PLAYER_COLS]
        out[m] = tuple((mods[k].predict_proba(X)[:, 1].astype(np.float32) for k in ('in', 'out')))
        print('  P(borde de parada) fuera', m, flush=True)
    np.savez_compressed(path, **{m + '_' + k: v for m, p in out.items() for k, v in zip(('in', 'out'), p)})
    return out

def pbound_barca(w=10):
    mods = _bd_models(matches(), w)
    X = V17.barca_features()[PLAYER_COLS]
    return tuple((mods[k].predict_proba(X)[:, 1].astype(np.float32) for k in ('in', 'out')))

def pchange_barca(cfg):
    import lightgbm as lgb
    rows = []
    for m in matches():
        for sd in cfg.get('pch_seeds', SEEDS):
            X = frames(m, sd, cfg.get('extra', ())).iloc[::STEP].copy()
            X['y'] = change_target(reference(m)[0])[::STEP]
            rows.append(X[X['label'] != 'D'])
    T = pd.concat(rows, ignore_index=True)
    cols = [c for c in V17.feature_cols(T) if c != 'y']
    mdl = lgb.LGBMClassifier(**V17.LGB_PARAMS)
    mdl.fit(T[cols], T['y'].astype(int))
    XB = barca_X(cfg)
    return (mdl.predict_proba(XB[cols])[:, 1].astype(np.float32), XB)

def barca_X(cfg):
    ex = cfg.get('extra', ())
    base = [barca_bc_features()] if 'bc' in ex else [V17.barca_features(), CF.barca_chain()]
    return pd.concat(base + [BARCA_EXTRA[g]() for g in ex if g != 'bc'], axis=1)

def barca_state(e, name):
    from src.io import load_atd
    cfg = e['cfg']
    pool = matches() if cfg.get('metrica_train', False) else EI.list_matches()
    models, cols, _ = fit(pool, cfg)
    pch, XB = pchange_barca(cfg) if e.get('alpha') or e.get('snap_w') else (None, None)
    if XB is None:
        XB = barca_X(cfg)
    p_dead, p_home = _predict_any(models, XB, cols)
    rules = pd.read_csv(V20.RULES_CSV, keep_default_na=False).label.values
    sig = XB[['last_ctrl', 'since_ctrl', 'ball_vis']].reset_index(drop=True)
    probs = (rules == 'D', p_home, p_dead)
    if e.get('dead_src') == 'players10':
        pdp = _pd_players_model(matches()).predict_proba(V17.barca_features()[PLAYER_COLS])[:, 1].astype(np.float32)
        probs = probs + (sig, pdp)
    pbd = pbound_barca(e['dead_var'].get('w', 10)) if e.get('dead_var') else None
    lab = full_labels('barca', probs, sig, pch, e, pbd=pbd)
    lab = V17.apply_definitions(lab, load_atd())
    out = os.path.join(OUT_DIR, 'barca_{}.csv'.format(name))
    pd.DataFrame({'frame': np.arange(len(lab)), 'label': lab}).to_csv(out, index=False)
    return out
if __name__ == '__main__':
    main()
