import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline_approach_v21 as P
from src import spell_select as SS
COSTS = (5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 135.0)
CACHE = {}
OVERRIDE = {}

def seed_path(sd):
    e = state()['e']
    tag = e['oof']
    pt = e.get('pch_tag', tag)
    if pt not in (tag, 'C_v5'):
        tag = tag + '_' + pt
    if e.get('seed_tag'):
        tag = tag + '_' + e['seed_tag']
    name = 'noche_seed{}.npz'.format(sd) if tag == 'C_v5' else 'noche_seed{}_{}.npz'.format(sd, tag)
    return os.path.join(P.OUT_DIR, name)

def state():
    if CACHE:
        return CACHE
    e = P.estado()
    e.update(OVERRIDE)
    P.setup(int(e['cfg'].get('deg', 2)))
    oof = P.run_oof(e['cfg'], e['oof'])
    sig = {m: P.signals(m) for m in oof}
    pch = P.run_pchange(e['cfg'], e.get('pch_tag', e['oof']))
    pdp = P.run_pd_players(e['cfg'].get('pd_seeds'), bool(e['cfg'].get('pd_long', False)), int(e['cfg'].get('pd_restart', 0)))
    oof = {m: oof[m][:3] + (sig[m], pdp[m]) for m in oof}
    kick = {m: P.kick_of(m) for m in oof}
    CACHE.update(e=e, oof=oof, sig=sig, pch=pch, kick=kick)
    return CACHE

def decodings(m, costs=COSTS, extra=None):
    st = state()
    e = st['e']
    out = {}
    for c in costs:
        dd = {'c_team': float(c)}
        if extra:
            dd.update(extra)
        out['c{:g}'.format(c)] = P.full_labels(m, st['oof'][m], st['sig'][m], st['pch'][m], e, dd, kick=st['kick'][m])
    return out

def base_labels(m):
    st = state()
    return P.full_labels(m, st['oof'][m], st['sig'][m], st['pch'][m], st['e'], kick=st['kick'][m])
LGB_SEL = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0, n_jobs=8, deterministic=True, force_row_wise=True)
LAMBDAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

def pdead_mix(e, probs):
    from src import decoding_v21 as DV
    p = DV.rule_pdead(probs[0])
    if e.get('w_rule', 1.0) < 1.0 and len(probs) >= 5:
        p = e['w_rule'] * p + (1 - e['w_rule']) * probs[4]
    return p

def taker_of(mid, seed=0):
    from src.restart_features import taker_series
    if mid == 'barca':
        from src.io import load_atd
        from src import config as C
        path = os.path.join(P.V17.FEAT_DIR, 'barca_taker.parquet')
        if not os.path.isfile(path):
            pd.DataFrame({'tk': taker_series(load_atd(), C.HALVES, (C.PITCH_LENGTH, C.PITCH_WIDTH))}).to_parquet(path)
    else:
        path = os.path.join(P.V17.FEAT_DIR, '{}_{}{}_taker.parquet'.format(mid, P.EI.deg_tag(), seed))
        if not os.path.isfile(path):
            m = P.EM.load_degraded(mid, seed) if P.is_metrica(mid) else P.EI.load_degraded(mid, seed)
            pd.DataFrame({'tk': taker_series(m, P.V17.ext_halves(m), P.EM.PITCH if P.is_metrica(mid) else P.V17.EXT_PITCH)}).to_parquet(path)
    return pd.read_parquet(path)['tk'].values

def cand_table(labs, probs, sig, pch, kick, e, order, extra=False, F=None, tk=None):
    S, E, T, cnt, origin = SS.candidates(labs)
    if SHIFTS['on']:
        S, E, T, cnt, origin = SS.add_shifts(S, E, T, cnt, origin, len(kick))
    vis = (sig['ball_vis'].values >= 0.5).astype(float)
    ctrl = np.where(sig['since_ctrl'].values == 0, sig['last_ctrl'].values, 0.0)
    X = SS.features(S, E, T, cnt, origin, labs, np.asarray(probs[1], float), pdead_mix(e, probs), np.asarray(pch, float), vis, ctrl, np.asarray(kick, float), order, extra=extra)
    if extra == 3 and F is not None:
        X = pd.concat([X.reset_index(drop=True), SS.frame_features(S, E, T, F.reset_index(drop=True))], axis=1)
    if extra in (4, 5) and tk is not None:
        X = pd.concat([X.reset_index(drop=True), SS.taker_features(S, E, T, np.asarray(tk))], axis=1)
    if extra == 5 and F is not None:
        X = pd.concat([X.reset_index(drop=True), SS.restart_edge_features(S, E, F.reset_index(drop=True))], axis=1)
    return (S, E, T, X)
ALT = {'metrica': ('D_v5_metrica', {'metrica_train': True}), 'ctrl': ('N_v5_controlada', {'idsse_label': 'controlada'})}

def alt_probs(m, key):
    k = ('alt', key, m)
    if k not in CACHE:
        tag, over = ALT[key]
        cfg = dict(state()['e']['cfg'], **over)
        if m == 'barca':
            pool = P.matches() if cfg.get('metrica_train', False) else P.EI.list_matches()
            models, cols, _ = P.fit(pool, cfg)
            p_dead, p_home = P._predict_any(models, P.barca_X(cfg), cols)
            rules = pd.read_csv(P.V20.RULES_CSV, keep_default_na=False).label.values
            CACHE[k] = (rules == 'D', p_home, p_dead)
        else:
            CACHE[k] = P.run_oof(cfg, tag)[m]
    return CACHE[k]

def spec_labels(m, specs, probs=None, sig=None, pch=None, kick=None):
    st = state()
    e = st['e']
    if probs is None:
        probs, sig, pch, kick = (st['oof'][m], st['sig'][m], st['pch'][m], st['kick'][m])
    out = {}
    for nm, dd, eo in specs:
        pr = probs
        eo = dict(eo)
        if '_short' in eo:
            base, thr = eo['_short']
            out[nm] = SS.insert_short(out[base], pch, thr)
            continue
        if '_oof' in eo:
            pr = tuple(alt_probs(m, eo.pop('_oof'))[:3]) + tuple(probs[3:])
        out[nm] = P.full_labels(m, pr, sig, pch, dict(e, **eo), dd, kick=kick)
    return out

def short_specs(base_costs=(15.0, 30.0, 60.0), thrs=None):
    thrs = thrs or tuple((float(t) for t in os.environ.get('SEQ_SHORT_THRS', '0.2,0.3').split(',')))
    'Decodificaciones derivadas con posesiones cortas insertadas entre picos de P(cambio) (ver SS.insert_short).'
    return [('c{:g}_short{:g}'.format(c, t), {}, {'_short': ('c{:g}'.format(c), t)}) for c in base_costs for t in thrs]

def alt_specs(key):
    return [('{}_c{:g}'.format(key, c), {'c_team': c}, {'_oof': key}) for c in (10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0)] + [('{}_c30_d{:g}'.format(key, cd), {'c_team': 30.0, 'c_dead': cd}, {'_oof': key}) for cd in (10.0, 40.0)]

def cost_specs(costs=COSTS):
    return [('c{:g}'.format(c), {'c_team': float(c)}, {}) for c in costs]

def diverse_specs():
    sp = cost_specs()
    for cd in (10.0, 40.0):
        for c in (15.0, 30.0, 60.0):
            sp.append(('c{:g}_d{:g}'.format(c, cd), {'c_team': c, 'c_dead': cd}, {}))
    for w in (0.1, 0.6):
        for c in (15.0, 30.0, 60.0):
            sp.append(('c{:g}_w{:g}'.format(c, w), {'c_team': c}, {'w_rule': w}))
    return sp
SHIFTS = {'on': False}

def more_specs():
    sp = diverse_specs()
    for a in (0.0, 0.5):
        for c in (15.0, 30.0, 60.0):
            sp.append(('c{:g}_a{:g}'.format(c, a), {'c_team': c}, {'alpha': a}))
    for mn in (1.0, 2.0):
        sp.append(('c30_m{:g}'.format(mn), {'c_team': 30.0, 'min_s': mn}, {}))
    for c in (3.0, 200.0, 300.0):
        sp.append(('c{:g}'.format(c), {'c_team': c}, {}))
    sp.append(('c30_nosnap', {'c_team': 30.0}, {'restart_snap': None}))
    sp.append(('c30_rules', {'c_team': 30.0, 'c_dead': 80.0}, {'w_rule': 1.0}))
    return sp

def spec_tables(specs, extra=False):
    key = ('spec', tuple((s_[0] for s_ in specs)), extra, SHIFTS['on'])
    if key in CACHE:
        return CACHE[key]
    st = state()
    out = {}
    order = [s_[0] for s_ in specs]
    for m in st['oof']:
        labs = spec_labels(m, specs)
        S, E, T, X = cand_table(labs, st['oof'][m], st['sig'][m], st['pch'][m], st['kick'][m], st['e'], order, extra, F=P.frames(m, 0) if extra == 3 else P.restart_frame(m, 0, 2) if extra == 5 else None, tk=taker_of(m, 0) if extra in (4, 5) else None)
        out[m] = (S, E, T, X, SS.match_value(SS.best_iou(S, E, T, P.reference(m)[0])))
    CACHE[key] = out
    return out

def ext_tables(costs=COSTS):
    key = ('tables', costs)
    if key in CACHE:
        return CACHE[key]
    st = state()
    out = {}
    order = ['c{:g}'.format(c) for c in costs]
    for m in st['oof']:
        labs = decodings(m, costs)
        S, E, T, X = cand_table(labs, st['oof'][m], st['sig'][m], st['pch'][m], st['kick'][m], st['e'], order)
        ref = P.reference(m)[0]
        out[m] = (S, E, T, X, SS.match_value(SS.best_iou(S, E, T, ref)))
    CACHE[key] = out
    return out
SCORER = {'obj': 'l2', 'seeds': 1}

class _Bag:

    def __init__(self, ms):
        self.ms = ms

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.ms], axis=0)

def fit_scorer(tabs, mids, cols=None):
    import lightgbm as lgb
    X = pd.concat([tabs[m][3] for m in mids], ignore_index=True)
    y = np.concatenate([tabs[m][4] for m in mids])
    cols = cols or list(X.columns)
    ms = []
    for sd in range(SCORER['seeds']):
        prm = dict(LGB_SEL, random_state=sd)
        if SCORER['obj'] == 'xent':
            prm['objective'] = 'cross_entropy'
        mdl = lgb.LGBMRegressor(**prm)
        mdl.fit(X[cols], y)
        ms.append(mdl)
    return (_Bag(ms), cols)

def oof_scores(tabs, cols=None):
    out = {}
    for m in tabs:
        mdl, c = fit_scorer(tabs, [k for k in tabs if k != m], cols)
        out[m] = np.clip(mdl.predict(tabs[m][3][c]), 0, 1)
    return out

def eval_select(tabs, p, lambdas=LAMBDAS):
    cands = []
    for lam in lambdas:
        per = {}
        for m, (S, E, T, X, v) in tabs.items():
            idx = SS.select(S, E, 2 * p[m] - lam)
            per[m] = P.score_match(m, SS.render(len(P.reference(m)[0]), S, E, T, idx))
        cands.append(({'lam': lam}, per))
    d, per = max(cands, key=lambda c: P.nota(c[1])[0])
    return (d['lam'], per, P.nested_choice(cands), cands)

def base_per():
    st = state()
    return {m: P.score_match(m, base_labels(m)) for m in st['oof']}

def barca_inputs():
    if 'barca' in CACHE:
        return CACHE['barca']
    e = state()['e']
    cfg = e['cfg']
    pool = P.matches() if cfg.get('metrica_train', False) else P.EI.list_matches()
    models, cols, _ = P.fit(pool, cfg)
    pch, XB = P.pchange_barca(cfg)
    p_dead, p_home = P._predict_any(models, XB, cols)
    rules = pd.read_csv(P.V20.RULES_CSV, keep_default_na=False).label.values
    sig = XB[['last_ctrl', 'since_ctrl', 'ball_vis']].reset_index(drop=True)
    pdp = P.pd_players_predict(P._pd_players_model(P.matches(), cfg.get('pd_seeds'), bool(cfg.get('pd_long', False)), int(cfg.get('pd_restart', 0))), 'barca', 0)
    probs = (rules == 'D', p_home, p_dead, sig, pdp)
    CACHE['barca'] = (probs, sig, pch, P.kick_of('barca'))
    return CACHE['barca']

def barca_select(tabs, lam, name, costs=COSTS, cols=None, specs=None, extra=False):
    from src.io import load_atd
    e = state()['e']
    probs, sig, pch, kick = barca_inputs()
    specs = specs or cost_specs(costs)
    order = [s_[0] for s_ in specs]
    labs = spec_labels('barca', specs, probs, sig, pch, kick)
    S, E, T, X = cand_table(labs, probs, sig, pch, kick, e, order, extra, F=P.barca_X(e['cfg']) if extra == 3 else P.restart_frame('barca', 0, 2) if extra == 5 else None, tk=taker_of('barca') if extra in (4, 5) else None)
    mdl, c = fit_scorer(tabs, list(tabs), cols)
    p = np.clip(mdl.predict(X[c]), 0, 1)
    lab = SS.render(len(kick), S, E, T, SS.select(S, E, 2 * p - lam))
    lab = P.V17.apply_definitions(lab, load_atd())
    out = os.path.join(P.OUT_DIR, 'barca_{}.csv'.format(name))
    pd.DataFrame({'frame': np.arange(len(lab)), 'label': lab}).to_csv(out, index=False)
    print(P.compare_barca(out).to_string(index=False), flush=True)
    return out
LAMBDAS2 = (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0)

def specs_of(name):
    base = {'costes': cost_specs, 'variedad': diverse_specs, 'mas': more_specs}
    parts = name.split('+')
    sp = base[parts[0]]()
    for p_ in parts[1:]:
        sp = sp + (short_specs() if p_ == 'cortas' else alt_specs(p_))
    return sp

def compute_seeds(seeds):
    from src import decoding_v21 as DV
    import lightgbm as lgb
    todo = [sd for sd in seeds if not os.path.isfile(seed_path(sd))]
    if not todo:
        return
    st = state()
    cfg = st['e']['cfg']
    mids = list(st['oof'])
    pool = P.matches() if cfg.get('metrica_train', False) else P.EI.list_matches()
    rows = {}
    for m in mids:
        for sd in cfg.get('pch_seeds', P.SEEDS):
            X = P.frames(m, sd, cfg.get('extra', ())).iloc[::P.STEP].copy()
            X['y'] = P.change_target(P.reference(m)[0])[::P.STEP]
            rows[m, sd] = X
    pcols = [c for c in P.V17.feature_cols(next(iter(rows.values()))) if c != 'y']
    raw = {sd: {} for sd in todo}
    for m in mids:
        models, cols, _ = P.fit([k for k in pool if k != m], cfg)
        T = pd.concat([X for (k, _), X in rows.items() if k != m], ignore_index=True)
        T = T[T['label'] != 'D']
        mdl = lgb.LGBMClassifier(**P.V17.LGB_PARAMS)
        mdl.fit(T[pcols], T['y'].astype(int))
        pdm = P._pd_players_model([k for k in P.matches() if k != m], cfg.get('pd_seeds'), bool(cfg.get('pd_long', False)), int(cfg.get('pd_restart', 0)))
        for sd in todo:
            dead, ph, pdd = P.predict(models, cols, m, cfg, seed=sd)
            Xs = P.frames(m, sd, cfg.get('extra', ()))
            pch = mdl.predict_proba(Xs[pcols])[:, 1].astype(np.float32)
            pdp = P.pd_players_predict(pdm, m, sd)
            d = P.EM.load_degraded(m, sd) if P.is_metrica(m) else P.EI.load_degraded(m, sd)
            raw[sd][m] = (dead, ph, pdd, pch, pdp, DV.ball_kick(d['ball']))
            print('  semilla {} fuera {}'.format(sd, m), flush=True)
    for sd in todo:
        np.savez_compressed(seed_path(sd), **{m + k: v for m, r in raw[sd].items() for k, v in zip(('_dead', '_ph', '_pd', '_pch', '_pdp', '_kick'), r)})

def state_seed(seed=1):
    from src import decoding_v21 as DV
    import lightgbm as lgb
    key = ('seed', seed)
    if key in CACHE:
        return CACHE[key]
    path = seed_path(seed)
    st = state()
    e = st['e']
    cfg = e['cfg']
    mids = list(st['oof'])
    if os.path.isfile(path):
        z = np.load(path)
        raw = {m: tuple((z[m + k] for k in ('_dead', '_ph', '_pd', '_pch', '_pdp', '_kick'))) for m in mids}
    else:
        raw = {}
        pool = P.matches() if cfg.get('metrica_train', False) else P.EI.list_matches()
        rows = {}
        for m in mids:
            for sd in P.SEEDS:
                X = P.frames(m, sd, cfg.get('extra', ())).iloc[::P.STEP].copy()
                X['y'] = P.change_target(P.reference(m)[0])[::P.STEP]
                rows[m, sd] = X
        pcols = [c for c in P.V17.feature_cols(next(iter(rows.values()))) if c != 'y']
        for m in mids:
            models, cols, _ = P.fit([k for k in pool if k != m], cfg)
            dead, ph, pdd = P.predict(models, cols, m, cfg, seed=seed)
            T = pd.concat([X for (k, _), X in rows.items() if k != m], ignore_index=True)
            T = T[T['label'] != 'D']
            mdl = lgb.LGBMClassifier(**P.V17.LGB_PARAMS)
            mdl.fit(T[pcols], T['y'].astype(int))
            Xs = P.frames(m, seed, cfg.get('extra', ()))
            pch = mdl.predict_proba(Xs[pcols])[:, 1].astype(np.float32)
            pdp = P._pd_players_model([k for k in P.matches() if k != m]).predict_proba(Xs[P.PLAYER_COLS])[:, 1].astype(np.float32)
            d = P.EM.load_degraded(m, seed) if P.is_metrica(m) else P.EI.load_degraded(m, seed)
            raw[m] = (dead, ph, pdd, pch, pdp, DV.ball_kick(d['ball']))
            print('  semilla {} fuera {}'.format(seed, m), flush=True)
        np.savez_compressed(path, **{m + k: v for m, r in raw.items() for k, v in zip(('_dead', '_ph', '_pd', '_pch', '_pdp', '_kick'), r)})
    out = {}
    for m in mids:
        dead, ph, pdd, pch, pdp, kick = raw[m]
        sig = P.frames(m, seed)[['last_ctrl', 'since_ctrl', 'ball_vis']].reset_index(drop=True)
        out[m] = ((dead, ph, pdd, sig, pdp), sig, pch, kick)
    CACHE[key] = out
    return out

def seed_tables(specs, extra, seed=1):
    key = ('spec_seed', seed, tuple((s_[0] for s_ in specs)), extra, SHIFTS['on'])
    if key in CACHE:
        return CACHE[key]
    st = state()
    ss = state_seed(seed)
    out = {}
    order = [s_[0] for s_ in specs]
    for m in st['oof']:
        probs, sig, pch, kick = ss[m]
        labs = spec_labels(m, [s_ for s_ in specs if '_oof' not in s_[2]], probs, sig, pch, kick)
        S, E, T, X = cand_table(labs, probs, sig, pch, kick, st['e'], order, extra, F=P.frames(m, seed) if extra == 3 else P.restart_frame(m, seed, 2) if extra == 5 else None, tk=taker_of(m, seed) if extra in (4, 5) else None)
        out[m] = (S, E, T, X, SS.match_value(SS.best_iou(S, E, T, P.reference(m)[0])))
    CACHE[key] = out
    return out

def oof_scores2(tabs, tabs1):
    extra = tabs1 if isinstance(tabs1, list) else [tabs1]
    out = {}
    for m in tabs:
        others = [k for k in tabs if k != m]
        both = {k: tabs[k] for k in others}
        for j, tb in enumerate(extra):
            both.update({k + '#{}'.format(j + 1): tb[k] for k in others})
        mdl, c = fit_scorer(both, list(both))
        out[m] = np.clip(mdl.predict(tabs[m][3][c]), 0, 1)
    return out

def select_adaptive(S, E, p, a, iters=8):
    lam = 0.8
    idx = SS.select(S, E, 2 * p - lam)
    for _ in range(iters):
        f = float(np.mean(p[idx])) if len(idx) else 0.5
        lam_new = a * f
        if abs(lam_new - lam) < 0.001:
            break
        lam = lam_new
        idx = SS.select(S, E, 2 * p - lam)
    return (idx, lam)

def _two_pass(tabs, tabs1, lam1):

    def first(m):
        others = [k for k in tabs if k != m]
        both = {k: tabs[k] for k in others}
        both.update({k + '#1': tabs1[k] for k in others})
        mdl, c = fit_scorer(both, list(both))
        return (np.clip(mdl.predict(tabs[m][3][c]), 0, 1), np.clip(mdl.predict(tabs1[m][3][c]), 0, 1))
    ctx, ctx1 = ({}, {})
    for m in tabs:
        p, p1 = first(m)
        for tb, pp, dst in ((tabs, p, ctx), (tabs1, p1, ctx1)):
            S, E, T, X, v = tb[m]
            idx = SS.select(S, E, 2 * pp - lam1)
            dst[m] = (S, E, T, pd.concat([X.reset_index(drop=True), SS.context_features(S, E, T, pp, idx)], axis=1), v)
    return (oof_scores2(ctx, ctx1), ctx, ctx1)

def state_ref():
    e = P.estado()
    sel = e['spell_select']
    SCORER.update(obj=sel.get('obj', 'l2'), seeds=int(sel.get('seeds', 1)))
    SHIFTS['on'] = bool(sel.get('shifts', False))
    SEED1['on'] = bool(sel.get('seed1', False))
    SEED1['seeds'] = list(sel.get('extra_seeds', [1]))
    specs = specs_of(sel['specs'])
    tabs = spec_tables(specs, sel.get('extra', False))
    lam, per, nest, _ = eval_select(tabs, scores_for(tabs, specs, sel.get('extra', False)), LAMBDAS2)
    SHIFTS['on'] = False
    print('estado: λ {} {:.4f} anidada {:.4f}'.format(lam, P.nota(per)[0], nest), flush=True)
    return (per, nest)
SEED1 = {'on': False, 'seeds': [1]}

def scores_for(tabs, specs, extra):
    if SEED1['on']:
        compute_seeds(SEED1['seeds'])
        return oof_scores2(tabs, [seed_tables(specs, extra, sd) for sd in SEED1['seeds']])
    return oof_scores(tabs)

def train_tabs(tabs, specs, extra):
    if SEED1['on']:
        compute_seeds(SEED1['seeds'])
        out = dict(tabs)
        for j, sd in enumerate(SEED1['seeds']):
            out.update({k + '#{}'.format(j + 1): v for k, v in seed_tables(specs, extra, sd).items()})
        return out
    return tabs

def final_select(name='final', estado_file=None):
    if estado_file:
        e = json.load(open(os.path.join(P.OUT_DIR, estado_file)))
        CACHE.clear()
        OVERRIDE.clear()
        OVERRIDE.update({k: e.get(k) for k in ('cfg', 'oof', 'seed_tag')})
    else:
        e = P.estado()
    sel = e['spell_select']
    SCORER.update(obj=sel.get('obj', 'l2'), seeds=int(sel.get('seeds', 1)))
    SHIFTS['on'] = bool(sel.get('shifts', False))
    SEED1['on'] = bool(sel.get('seed1', False))
    SEED1['seeds'] = list(sel.get('extra_seeds', [1]))
    specs = specs_of(sel['specs'])
    tabs = spec_tables(specs, sel.get('extra', False))
    return barca_select(train_tabs(tabs, specs, sel.get('extra', False)), float(sel['lam']), name, specs=specs, extra=sel.get('extra', False))
STEPS = {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True)
    a = ap.parse_args()
    t = time.time()
    STEPS[a.step]()
    print('hecho en {:.0f} s'.format(time.time() - t), flush=True)
if __name__ == '__main__':
    main()
