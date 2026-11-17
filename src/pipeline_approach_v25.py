import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline_approach_v21 as P
import noche_v21 as N
from src import spell_select as SS
from src.spell_decoding import runs as _runs
NAME = 'pipeline_approach_v25'
V21_DIR = P.OUT_DIR
OUT_DIR = os.path.join(P.C.ROOT, 'experiments', NAME)
P.NAME, P.OUT_DIR = (NAME, OUT_DIR)
P.PASOS_CSV = os.path.join(OUT_DIR, 'pasos.csv')
P.ESTADO = os.path.join(OUT_DIR, 'estado.json')
V21_SEL = os.path.join(V21_DIR, 'ours_pipeline_approach_v21_selector.csv')
V24 = os.path.join(P.C.ROOT, 'experiments', 'pipeline_approach_v24', 'ours_pipeline_approach_v24.csv')
REUSE = ('oof_C_v5.npz', 'pchange_C_v5.npz', 'pd_players10.npz', 'noche_seed1.npz')
TMP = os.path.join(P.C.ROOT, 'cache', 'v25')
FPS = 25
LAMS = N.LAMBDAS2

def init():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    if not os.path.isfile(P.ESTADO):
        shutil.copy(os.path.join(V21_DIR, 'estado_N1.5.json'), P.ESTADO)
    for f in REUSE:
        src, dst = (os.path.join(V21_DIR, f), os.path.join(OUT_DIR, f))
        if os.path.isfile(src) and (not os.path.isfile(dst)):
            shutil.copy(src, dst)

def _sel_setup():
    e = P.estado()
    sel = e['spell_select']
    N.SCORER.update(obj=sel.get('obj', 'l2'), seeds=int(sel.get('seeds', 1)))
    N.SHIFTS['on'] = False
    N.SEED1['on'] = bool(sel.get('seed1', False))
    N.SEED1['seeds'] = [1]
    return (e, N.specs_of(sel['specs']), sel.get('extra', False))

def base(tag='base', force=False):
    path = os.path.join(TMP, tag + '.pkl')
    if os.path.isfile(path) and (not force):
        return pickle.load(open(path, 'rb'))
    e, specs, extra = _sel_setup()
    tabs = N.spec_tables(specs, extra)
    p = N.scores_for(tabs, specs, extra)
    labs = {m: {lam: SS.render(len(P.reference(m)[0]), S, E, T, SS.select(S, E, 2 * p[m] - lam)) for lam in LAMS} for m, (S, E, T, X, v) in tabs.items()}
    out = {'p': p, 'labs': labs, 'tabs': {m: t[:3] + (None, t[4]) for m, t in tabs.items()}}
    pickle.dump(out, open(path, 'wb'))
    return out
_CTX = {}

def ctx(m):
    if m in _CTX:
        return _CTX[m]
    e = P.estado()
    if m == 'barca':
        from src import config as C
        X = P.barca_X(e['cfg']).reset_index(drop=True)
        halves = dict(C.HALVES)
    else:
        P.setup(int(e['cfg'].get('deg', 5)))
        X = P.frames(m, 0).reset_index(drop=True)
        d = P.EM.load_degraded(m, 0) if P.is_metrica(m) else P.EI.load_degraded(m, 0)
        halves = P.V17.ext_halves(d)
    c = {'n': len(X), 'halves': halves}
    for k in ('d1_h', 'd1_a', 'ball_vis', 'ball_speed', 'ball_out', 'ball_d_side', 'ball_d_end', 'last_ctrl', 'since_ctrl', 'ch_v15_d', 'p90', 'gap_len', 'gap_move', 'n_out_players', 'bspeed_max_f2', 'bspeed_max_p2', 'ball_att_h', 'ball_att_a'):
        c[k] = X[k].values.astype(float)
    c['dead'] = c['ch_v15_d'] > 0.5
    if m == 'barca':
        from src.io import load_atd
        c['ball_xy'] = np.asarray(load_atd()['ball'], float)[:, :2]
    else:
        c['ball_xy'] = np.asarray(d['ball'], float)[:, :2]
    pth = os.path.join(TMP, 'ph_{}.npy'.format(m))
    if m == 'barca':
        if not os.path.isfile(pth):
            np.save(pth, np.asarray(N.barca_inputs()[0][1], float))
        c['ph'] = np.load(pth)
    else:
        c['ph'] = np.load(os.path.join(OUT_DIR, 'oof_C_v5.npz'))[m + '_ph'].astype(float)
    _CTX[m] = c
    return c

def spans(lab, a, b):
    st, en = _runs(np.asarray(lab[a:b]))
    return [(int(s) + a, int(t) + a, str(lab[s + a])) for s, t in zip(st, en)]

def r_control(lab, c, radius=1.5, min_s=0.4, min_visible=0.5):
    out = lab.copy()
    need = int(round(min_s * FPS))
    vis = c['ball_vis'] >= 0.5
    for a, b in c['halves'].values():
        rr = spans(out, a, b + 1)
        for i in range(1, len(rr)):
            prev = out[rr[i][0] - 1]
            s, e, cur = rr[i]
            if cur not in 'HA' or prev not in 'HA' or prev == cur:
                continue
            if vis[s:e].mean() < min_visible:
                continue
            d = (c['d1_h'] if cur == 'H' else c['d1_a'])[s:e]
            with np.errstate(invalid='ignore'):
                close = np.r_[0, (d < radius).astype(np.int8), 0]
            st, en = (np.flatnonzero(np.diff(close) == 1), np.flatnonzero(np.diff(close) == -1))
            if (en - st).max(initial=0) < need:
                out[s:e] = prev
    return out

def _reason(c, s, e, thr):
    return float(np.mean(c['ch_v15_d'][s:e] > 0.5)) >= thr

def r_fill(lab, c, same_s=0.0, diff_s=0.0, reason=1.01):
    out = lab.copy()
    for a, b in c['halves'].values():
        rr = spans(lab, a, b + 1)
        for i in range(1, len(rr) - 1):
            s, e, v = rr[i]
            if v != 'D':
                continue
            pv, nv = (rr[i - 1][2], rr[i + 1][2])
            if pv not in 'HA' or nv not in 'HA':
                continue
            L = (e - s) / FPS
            if reason <= 1 and _reason(c, s, e, reason):
                continue
            if pv == nv and L <= same_s:
                out[s:e] = pv
            elif pv != nv and L <= diff_s:
                mid = (s + e) // 2
                out[s:mid], out[mid:e] = (pv, nv)
    return out

def r_shift(lab, c, d_start=0, d_end=0):
    if not d_start and (not d_end):
        return lab
    out = np.full(len(lab), 'D', dtype='<U1')
    for a, b in c['halves'].values():
        rr = spans(lab, a, b + 1)
        for i, (s, e, v) in enumerate(rr):
            if v not in 'HA':
                continue
            s2 = s + d_start if i == 0 or rr[i - 1][2] == 'D' else s
            e2 = e + d_end if i == len(rr) - 1 or rr[i + 1][2] == 'D' else e
            s2, e2 = (max(s2, a), min(e2, b + 1))
            if e2 - s2 >= 1:
                out[s2:e2] = np.where(out[s2:e2] == 'D', v, out[s2:e2])
    for a, b in c['halves'].values():
        for i, (s, e, v) in enumerate(spans(lab, a, b + 1)):
            if v in 'HA' and s > a and (out[s - 1] == v) and (lab[s - 1] != v):
                out[s] = 'D'
    return out

def r_snap(lab, c, w_s=1.0):
    out = lab.copy()
    w = int(round(w_s * FPS))
    ctrl = np.where(c['since_ctrl'] == 0, c['last_ctrl'], 0.0)
    for a, b in c['halves'].values():
        rr = spans(lab, a, b + 1)
        for i in range(1, len(rr)):
            s, e, v = rr[i]
            ps, pe, pv = rr[i - 1]
            if v not in 'HA' or pv not in 'HA' or v == pv:
                continue
            sign = 1.0 if v == 'H' else -1.0
            lo, hi = (max(ps + 1, s - w), min(e - 1, s + w))
            k = np.flatnonzero(ctrl[lo:hi] == sign)
            if len(k) == 0:
                continue
            t = lo + int(k[np.argmin(np.abs(k + lo - s))])
            if t < s:
                out[t:s] = v
            else:
                out[s:t] = pv
    return out

def r_trim(lab, c, pre_s=0.0, post_s=0.0):
    out = lab.copy()
    dead = c['dead']
    for a, b in c['halves'].values():
        rr = spans(lab, a, b + 1)
        for i in range(len(rr)):
            s, e, v = rr[i]
            if v != 'D':
                continue
            k = np.flatnonzero(dead[s:e])
            if len(k) == 0:
                continue
            f1, f2 = (s + int(k[0]), s + int(k[-1]) + 1)
            if i > 0 and rr[i - 1][2] in 'HA' and (0 < f1 - s <= pre_s * FPS):
                out[s:f1] = rr[i - 1][2]
            if i < len(rr) - 1 and rr[i + 1][2] in 'HA' and (0 < e - f2 <= post_s * FPS):
                out[f2:e] = rr[i + 1][2]
    return out

def r_fillph(lab, c, max_s=5.0, reason=0.3, min_s=1.0):
    out = lab.copy()
    for a, b in c['halves'].values():
        rr = spans(lab, a, b + 1)
        for i in range(1, len(rr) - 1):
            s, e, v = rr[i]
            if v != 'D' or rr[i - 1][2] not in 'HA' or rr[i + 1][2] not in 'HA':
                continue
            if e - s > max_s * FPS or _reason(c, s, e, reason):
                continue
            pv, nv = (rr[i - 1][2], rr[i + 1][2])
            if pv == nv:
                if e - s <= 2 * FPS:
                    out[s:e] = pv
                continue
            seg = np.where(c['ph'][s:e] > 0.5, 'H', 'A')
            ok = (seg == pv).astype(int)
            cs = np.r_[0, np.cumsum(ok)]
            tot = cs[-1]
            k = np.arange(e - s + 1)
            score = cs[k] + (e - s - k - (tot - cs[k]))
            cut = s + int(np.argmax(score))
            out[s:cut], out[cut:e] = (pv, nv)
    return out
BOX_ATT, BOX_SIDE = (-(1 - 16.5 / 52.5), 34.0 - 20.16)

def r_keeper(lab, c, max_s=8.0, look_s=1.0):
    out = lab.copy()
    vis = c['ball_vis'] >= 0.5
    for a, b in c['halves'].values():
        rr = spans(lab, a, b + 1)
        for i in range(1, len(rr) - 1):
            s, e, v = rr[i]
            nv = rr[i + 1][2]
            if v != 'D' or nv not in 'HA' or e - s > max_s * FPS:
                continue
            if np.nansum(c['ball_out'][s:e]) > 0:
                continue
            k = np.flatnonzero(vis[max(s - int(look_s * FPS), a):min(s + 13, e)])
            if len(k) == 0:
                continue
            f = max(s - int(look_s * FPS), a) + int(k[-1])
            att = c['ball_att_h' if nv == 'H' else 'ball_att_a'][f]
            if att < BOX_ATT and c['ball_d_side'][f] > BOX_SIDE:
                out[s:e] = nv
    return out

def r_outtrim(lab, c, max_s=2.0):
    out = lab.copy()
    inpl = (c['ball_vis'] >= 0.5) & (np.nan_to_num(c['ball_out']) < 0.5) & ~c['dead']
    for a, b in c['halves'].values():
        rr = spans(lab, a, b + 1)
        for i in range(1, len(rr)):
            s, e, v = rr[i]
            if v != 'D' or rr[i - 1][2] not in 'HA':
                continue
            k = np.flatnonzero(~inpl[s:e])
            f = s + (int(k[0]) if len(k) else e - s)
            if f < e and 0 < f - s <= max_s * FPS:
                out[s:f] = rr[i - 1][2]
    return out

def r_restart(lab, c, v_slow=5.0, d_opp=9.0, min_s=1.0, max_s=30.0, look_s=3.0):
    out = lab.copy()
    vis = c['ball_vis'] >= 0.5
    spd = np.nan_to_num(c['ball_speed'], nan=99.0)
    for a, b in c['halves'].values():
        rr = spans(lab, a, b + 1)
        for i in range(1, len(rr)):
            s, e, v = rr[i]
            if v not in 'HA' or rr[i - 1][2] != 'D':
                continue
            do = np.nan_to_num(c['d1_a' if v == 'H' else 'd1_h'][s:e], nan=0.0)
            far = vis[s:e] & (do >= d_opp)
            wait = far & (spd[s:e] < v_slow)
            dd = np.diff(np.r_[0, wait.astype(np.int8), 0])
            st, en = (np.flatnonzero(dd == 1), np.flatnonzero(dd == -1))
            for a0, b0 in zip(st, en):
                if a0 > look_s * FPS or not far[:a0].all():
                    break
                if b0 - a0 >= min_s * FPS:
                    n = min(int(b0), int(max_s * FPS))
                    if n < e - s:
                        out[s:s + n] = 'D'
                    break
    return out

def r_edge(lab, c, v_mov=3.0, p_run=3.0, max_s=6.0, side='both'):
    out = lab.copy()
    vis = c['ball_vis'] >= 0.5
    spd = np.nan_to_num(c['ball_speed'])
    play = (vis & (spd > v_mov) | (np.nan_to_num(c['p90']) > p_run)) & ~c['dead'] & (np.nan_to_num(c['ball_out']) < 0.5)
    w = int(max_s * FPS)
    for a, b in c['halves'].values():
        rr = spans(lab, a, b + 1)
        for i in range(1, len(rr) - 1):
            s, e, v = rr[i]
            if v != 'D':
                continue
            pv, nv = (rr[i - 1][2], rr[i + 1][2])
            if side in ('both', 'start') and pv in 'HA':
                k = np.flatnonzero(~play[s:min(e, s + w)])
                f = s + (int(k[0]) if len(k) else min(e, s + w) - s)
                out[s:f] = pv
            if side in ('both', 'end') and nv in 'HA':
                k = np.flatnonzero(~play[max(s, e - w):e][::-1])
                f = e - (int(k[0]) if len(k) else e - max(s, e - w))
                if f > s and (out[f - 1] == 'D' or f == e):
                    out[f:e] = nv
    return out

def r_gap(lab, c, min_gap_s=3.0, max_d=10.0):
    out = lab.copy()
    xy = c['ball_xy']
    miss = ~np.isfinite(xy[:, 0])
    st = np.r_[0, np.flatnonzero(miss[1:] != miss[:-1]) + 1]
    en = np.r_[st[1:], len(miss)]
    for a, b in zip(st, en):
        if not miss[a] or b - a < min_gap_s * FPS or a == 0 or (b >= len(miss)):
            continue
        dd = np.hypot(*xy[a - 1] - xy[b])
        if np.isfinite(dd) and dd < max_d:
            seg = out[a:b]
            seg[(seg == 'H') | (seg == 'A')] = 'D'
    return out

def r_falserestart(lab, c, v_still=1.0, still_s=1.0, look_s=6.0, touch_acc=22.0):
    import importlib.util
    spec = importlib.util.spec_from_file_location('fr', os.path.join(P.C.ROOT, 'tools', 'false_restart.py'))
    fr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fr)
    out, _, _ = fr.false_restarts(lab, np.c_[c['ball_xy'], np.zeros(len(lab))], v_still, still_s, look_s, touch_acc)
    return out
RULES = {'edge': r_edge, 'gap': r_gap, 'falserestart': r_falserestart, 'restart': r_restart, 'keeper': r_keeper, 'outtrim': r_outtrim, 'control': r_control, 'fill': r_fill, 'shift': r_shift, 'snap': r_snap, 'trim': r_trim, 'fillph': r_fillph}

def chain(lab, c, steps):
    for name, kw in steps:
        lab = RULES[name](lab, c, **kw)
    return lab

def barca_raw(lam, name='raw'):
    path = os.path.join(TMP, 'barca_{}_{:g}.npy'.format(name, lam))
    if os.path.isfile(path):
        return np.load(path)
    e, specs, extra = _sel_setup()
    tabs = N.spec_tables(specs, extra)
    tt = N.train_tabs(tabs, specs, extra)
    st = N.state()['e']
    probs, sig, pch, kick = N.barca_inputs()
    order = [s_[0] for s_ in specs]
    labs = N.spec_labels('barca', specs, probs, sig, pch, kick)
    S, E, T, X = N.cand_table(labs, probs, sig, pch, kick, st, order, extra)
    mdl, cc = N.fit_scorer(tt, list(tt), None)
    pp = np.clip(mdl.predict(X[cc]), 0, 1)
    lab = SS.render(len(kick), S, E, T, SS.select(S, E, 2 * pp - lam))
    np.save(path, lab)
    return lab

def barca_cands():
    path = os.path.join(TMP, 'barca_cands.pkl')
    if os.path.isfile(path):
        return pickle.load(open(path, 'rb'))
    e, specs, extra = _sel_setup()
    tabs = N.spec_tables(specs, extra)
    tt = N.train_tabs(tabs, specs, extra)
    st = N.state()['e']
    probs, sig, pch, kick = N.barca_inputs()
    order = [s_[0] for s_ in specs]
    labs = N.spec_labels('barca', specs, probs, sig, pch, kick)
    S, E, T, X = N.cand_table(labs, probs, sig, pch, kick, st, order, extra)
    mdl, cc = N.fit_scorer(tt, list(tt), None)
    out = (S, E, T, np.clip(mdl.predict(X[cc]), 0, 1), len(kick))
    pickle.dump(out, open(path, 'wb'))
    return out

def qmap(p, pool):
    r = (np.argsort(np.argsort(p)) + 0.5) / len(p)
    return np.quantile(pool, r)

def consensus(S, E, T, p, thr=0.8):
    o = np.argsort(S)
    S2, E2, T2, p2 = (S[o], E[o], T[o], p[o])
    out = np.empty(len(p))
    for i in range(len(S2)):
        j0 = np.searchsorted(S2, S2[i] - (E2[i] - S2[i]), side='left')
        j1 = np.searchsorted(S2, E2[i], side='left')
        k = np.arange(j0, j1)
        k = k[T2[k] == T2[i]]
        iou = (np.minimum(E2[i], E2[k]) - np.maximum(S2[i], S2[k])) / (np.maximum(E2[i], E2[k]) - np.minimum(S2[i], S2[k]))
        out[i] = p2[k[iou >= thr]].mean()
    res = np.empty(len(p))
    res[o] = out
    return res
SELF = {'conf': 0.3, 'pd': 0.8, 'w': 3}

def pseudo_rows(models, cols, X, conf=None, pd_thr=None):
    conf = SELF['conf'] if conf is None else conf
    pd_thr = SELF['pd'] if pd_thr is None else pd_thr
    Xs = X.iloc[::P.STEP].copy()
    p_dead, p_home = P._predict_any(models, Xs, cols)
    rule_dead = Xs['ch_v15_d'].values > 0.5
    lab = np.full(len(Xs), '', dtype='<U1')
    lab[(p_dead > pd_thr) | rule_dead & (p_dead > 0.5)] = 'D'
    live = (p_dead < 1 - pd_thr) & ~rule_dead
    lab[live & (p_home > 0.5 + conf)] = 'H'
    lab[live & (p_home < 0.5 - conf)] = 'A'
    Xs['label'] = lab
    return Xs[lab != '']

def fit_self(train_mids, cfg, target_X):
    models, cols, dt = P.fit(train_mids, cfg)
    ps = pseudo_rows(models, cols, target_X)
    rows = P.train_rows([(m, s) for m in train_mids for s in cfg.get('train_seeds', P.SEEDS)], cfg)
    models2 = P.V17.fit_models(rows + [ps] * SELF['w'], cols, step=1)
    return (models2, cols, len(ps))

def oof_self(tag='V25_self'):
    path = P.oof_path(tag)
    if os.path.isfile(path):
        z = np.load(path)
        return {m: (z[m + '_dead'], z[m + '_ph'], z[m + '_pd']) for m in P.matches()}
    cfg = P.estado()['cfg']
    pool = P.matches() if cfg.get('metrica_train', False) else P.EI.list_matches()
    import gc
    out = {}
    for m in P.matches():
        part = os.path.join(TMP, '{}_{}.npz'.format(tag, m))
        if not os.path.isfile(part):
            X = P.frames(m, 0, cfg.get('extra', ()))
            models, cols, k = fit_self([x for x in pool if x != m], cfg, X)
            p_dead, p_home = P._predict_any(models, X, cols)
            np.savez_compressed(part, dead=X['ch_v15_d'].values > 0.5, ph=p_home, pd=p_dead)
            print('  fuera {} ({} pseudoetiquetas)'.format(m, k), flush=True)
            del models, X
            gc.collect()
        z = np.load(part)
        out[m] = (z['dead'], z['ph'], z['pd'])
    np.savez_compressed(path, **{m + k: v for m, pr in out.items() for k, v in zip(('_dead', '_ph', '_pd'), pr)})
    return out
HP = {'h1': dict(n_estimators=1000, learning_rate=0.03), 'h2': dict(num_leaves=127, min_child_samples=100), 'h3': dict(n_estimators=800, learning_rate=0.03, num_leaves=31, colsample_bytree=0.5), 'h4': dict(n_estimators=800, learning_rate=0.05, num_leaves=127, min_child_samples=400, reg_lambda=5.0), 'h5': dict(n_estimators=2000, learning_rate=0.02), 'h6': dict(n_estimators=1500, learning_rate=0.03, num_leaves=127, min_child_samples=400, reg_lambda=5.0)}

def oof_hp(tag, params):
    import gc
    cfg = P.estado()['cfg']
    pool = P.matches() if cfg.get('metrica_train', False) else P.EI.list_matches()
    keep = dict(P.V17.LGB_PARAMS)
    out = {}
    try:
        P.V17.LGB_PARAMS.update(params)
        for m in P.matches():
            part = os.path.join(TMP, 'hp_{}_{}.npz'.format(tag, m))
            if not os.path.isfile(part):
                models, cols, dt = P.fit([x for x in pool if x != m], cfg)
                X = P.frames(m, 0, cfg.get('extra', ()))
                p_dead, p_home = P._predict_any(models, X, cols)
                np.savez_compressed(part, dead=X['ch_v15_d'].values > 0.5, ph=p_home, pd=p_dead)
                print('  {} fuera {} ({:.0f} s)'.format(tag, m, dt), flush=True)
                del models, X
                gc.collect()
            z = np.load(part)
            out[m] = (z['dead'], z['ph'], z['pd'])
    finally:
        P.V17.LGB_PARAMS.clear()
        P.V17.LGB_PARAMS.update(keep)
    return out
_fit_prev = P.fit

def fit_lgb(train_mids, cfg):
    prm = cfg.get('lgb')
    if not prm:
        return _fit_prev(train_mids, cfg)
    keep = dict(P.V17.LGB_PARAMS)
    P.V17.LGB_PARAMS.update(prm)
    try:
        return _fit_prev(train_mids, cfg)
    finally:
        P.V17.LGB_PARAMS.clear()
        P.V17.LGB_PARAMS.update(keep)
P.fit = fit_lgb

def _run_max(x):
    d = np.diff(np.r_[0, x.astype(np.int8), 0])
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max(initial=0))

def new_feats(S, E, T, X):
    n = len(X)
    g = lambda k: np.nan_to_num(X[k].values.astype(float), nan=np.nan)
    dh, da = (g('d1_h'), g('d1_a'))
    vis = np.nan_to_num(g('ball_vis')) >= 0.5
    spd = np.nan_to_num(g('ball_speed'))
    dead = np.nan_to_num(g('ch_v15_d')) > 0.5
    bout = np.nan_to_num(g('ball_out')) > 0.5
    dside, dend = (g('ball_d_side'), g('ball_d_end'))
    ctrl = np.where(np.nan_to_num(g('since_ctrl'), nan=1) == 0, np.nan_to_num(g('last_ctrl')), 0.0)
    w1, w2 = (FPS, 2 * FPS)
    rows = []

    def mean(x, a, b):
        a, b = (max(a, 0), min(b, n))
        return float(np.mean(x[a:b])) if b > a else np.nan

    def nmin(x, a, b):
        a, b = (max(a, 0), min(b, n))
        v = x[a:b]
        v = v[np.isfinite(v)]
        return float(v.min()) if len(v) else np.nan
    for i in range(len(S)):
        a, b = (int(S[i]), int(E[i]))
        home = T[i] == 'H'
        do, dp = (dh, da) if home else (da, dh)
        sg = 1.0 if home else -1.0
        with np.errstate(invalid='ignore'):
            co, cp = (do[a:b] < 1.5, dp[a:b] < 1.5)
            closer = (do[a:b] < dp[a:b])[vis[a:b]]
        k = np.flatnonzero(ctrl[a:b] == sg)
        kb = np.flatnonzero(ctrl[max(a - 3 * FPS, 0):a] != 0)
        ka = np.flatnonzero(ctrl[b:min(b + 3 * FPS, n)] != 0)
        rows.append({'ca_own_run': _run_max(co) / FPS, 'ca_own_run2': _run_max(co[:w2]) / FPS, 'ca_opp_run': _run_max(cp) / FPS, 'ca_own_frac': float(co.mean()), 'ca_opp_frac': float(cp.mean()), 'ca_closer': float(closer.mean()) if len(closer) else np.nan, 'ca_d_own_start': nmin(do, a, a + w1), 'ca_d_opp_start': nmin(dp, a, a + w1), 'fl_frac': mean(spd >= 6.0, a, b), 'fl_start': mean(spd >= 6.0, a, a + w1), 'fl_end': mean(spd >= 6.0, b - w1, b), 'fl_before': mean(spd >= 6.0, a - w1, a), 'fl_max_start': float(spd[max(a - 12, 0):a + 13].max()), 'rs_dead_before': mean(dead, a - w2, a), 'rs_dead_after': mean(dead, b, b + w2), 'rs_dead_in': mean(dead, a, b), 'rs_out_before': mean(bout, a - w2, a), 'rs_out_after': mean(bout, b, b + w2), 'rs_side_before': nmin(dside, a - w2, a), 'rs_end_before': nmin(dend, a - w2, a), 'rs_side_after': nmin(dside, b, b + w2), 'rs_end_after': nmin(dend, b, b + w2), 'rs_invis_before': 1 - mean(vis, a - w2, a), 'rs_invis_after': 1 - mean(vis, b, b + w2), 'ec_first_own': k[0] / FPS if len(k) else 99.0, 'ec_last_own': (b - a - k[-1]) / FPS if len(k) else 99.0, 'ec_prev_ctrl': float(ctrl[max(a - 3 * FPS, 0):a][kb[-1]] * sg) if len(kb) else 0.0, 'ec_next_ctrl': float(ctrl[b:min(b + 3 * FPS, n)][ka[0]] * sg) if len(ka) else 0.0, 'ec_next_ctrl_t': ka[0] / FPS if len(ka) else 99.0})
    return pd.DataFrame(rows)
GROUPS = {'ca': 'control confirmado', 'fl': 'balón en vuelo', 'rs': 'motivo de parada y salida del balón', 'ec': 'controles en los bordes'}

def feat_tables(force=False):
    path = os.path.join(TMP, 'feat_tables.pkl')
    if os.path.isfile(path) and (not force):
        return pickle.load(open(path, 'rb'))
    e, specs, extra = _sel_setup()
    out = []
    for seed, tabs in ((0, N.spec_tables(specs, extra)), (1, N.seed_tables(specs, extra, 1))):
        d = {}
        for m, (S, E, T, X, v) in tabs.items():
            F = new_feats(S, E, T, P.frames(m, seed).reset_index(drop=True))
            d[m] = (S, E, T, pd.concat([X.reset_index(drop=True), F], axis=1), v)
            print('  variables', seed, m, len(S), flush=True)
        out.append(d)
    pickle.dump(out, open(path, 'wb'))
    return out

def _sub(tabs, prefixes):
    keep = lambda c: '_' not in c[:3] or c[:2] not in GROUPS or c[:2] in prefixes
    return {m: (S, E, T, X[[c for c in X.columns if not c[:3] in [g + '_' for g in GROUPS] or c[:2] in prefixes]], v) for m, (S, E, T, X, v) in tabs.items()}

def sel_labels(t0, t1):
    _sel_setup()
    p = N.oof_scores2(t0, [t1])
    labs = {m: {lam: SS.render(len(P.reference(m)[0]), S, E, T, SS.select(S, E, 2 * p[m] - lam)) for lam in LAMS} for m, (S, E, T, X, v) in t0.items()}
    return {'p': p, 'labs': labs}
SOURCES = ('v15', 'v13', 'v6', 'ctrl', 'close', 'ph')

def src_labels(X, ph, which=SOURCES):

    def lab(sign, dead):
        out = np.where(sign > 0, 'H', np.where(sign < 0, 'A', 'D')).astype('<U1')
        out[dead] = 'D'
        return out
    g = lambda k: np.nan_to_num(X[k].values.astype(float))
    dead = g('ch_v15_d') > 0.5
    allv = {'v15': lambda: lab(g('ch_v15_ha'), dead), 'v13': lambda: lab(g('ch_v13_ha'), g('ch_v13_d') > 0.5), 'v6': lambda: lab(g('ch_v6_ha'), g('ch_v6_d') > 0.5), 'ctrl': lambda: lab(g('last_ctrl'), dead), 'close': lambda: lab(g('last_close'), dead), 'ph': lambda: lab(np.asarray(ph, float) - 0.5, dead)}
    return {'src_' + k: allv[k]() for k in which}

def _table(m, seed, which, specs, extra, e):
    if m == 'barca':
        probs, sig, pch, kick = N.barca_inputs()
        F = P.barca_X(e['cfg']).reset_index(drop=True)
    elif seed == 0:
        st = N.state()
        probs, sig, pch, kick = (st['oof'][m], st['sig'][m], st['pch'][m], st['kick'][m])
        F = P.frames(m, 0).reset_index(drop=True)
    else:
        probs, sig, pch, kick = N.state_seed(seed)[m]
        F = P.frames(m, seed).reset_index(drop=True)
    order = [s_[0] for s_ in specs]
    labs = N.spec_labels(m, [s_ for s_ in specs if '_oof' not in s_[2]], probs, sig, pch, kick)
    labs.update(src_labels(F, probs[1], which))
    S, E, T, X = N.cand_table(labs, probs, sig, pch, kick, e, order, extra)
    S2, E2, T2, cnt, origin = SS.candidates(labs)
    assert len(S2) == len(S) and (S2 == S).all()
    X = X.reset_index(drop=True)
    for k in which:
        X['src_' + k] = [float('src_' + k in o) for o in origin]
    X['n_model'] = [float(sum((not x.startswith('src_') for x in o))) for o in origin]
    X = pd.concat([X, new_feats(S, E, T, F)], axis=1)
    v = None if m == 'barca' else SS.match_value(SS.best_iou(S, E, T, P.reference(m)[0]))
    return (S, E, T, X, v)

def src_tables(which=SOURCES, tag='src', force=False):
    path = os.path.join(TMP, '{}_tables.pkl'.format(tag))
    if os.path.isfile(path) and (not force):
        return pickle.load(open(path, 'rb'))
    e, specs, extra = _sel_setup()
    st = N.state()
    out = []
    for seed in (0, 1):
        d = {}
        for m in st['oof']:
            d[m] = _table(m, seed, which, specs, extra, st['e'])
            print('  tabla', seed, m, len(d[m][0]), flush=True)
        out.append(d)
    pickle.dump(out, open(path, 'wb'))
    return out

def _cols(tabs, drop_prefix=(), drop_cols=()):
    return {m: (S, E, T, X[[c for c in X.columns if not c.startswith(tuple(drop_prefix)) and c not in drop_cols]] if drop_prefix or drop_cols else X, v) for m, (S, E, T, X, v) in tabs.items()}

def _only_sources(tabs, which):
    out = {}
    drop = [k for k in SOURCES if k not in which]
    for m, (S, E, T, X, v) in tabs.items():
        keep = X['n_model'].values > 0
        for k in which:
            keep |= X['src_' + k].values > 0
        Xk = X[keep].drop(columns=['src_' + k for k in drop]).reset_index(drop=True)
        out[m] = (S[keep], E[keep], T[keep], Xk, v[keep])
    return out
NEWF = tuple((g + '_' for g in ('ca', 'fl', 'rs', 'ec')))

def final():
    from src.io import load_atd
    e = P.estado()
    P.setup(int(e['cfg'].get('deg', 2)))
    if not os.path.isfile(P.V20.RULES_CSV):
        P.V20.step_rules()
    lam = float(e['spell_select']['lam'])
    for f in ('barca_raw_{:g}.npy'.format(lam), 'barca_cands.pkl'):
        if os.path.isfile(os.path.join(TMP, f)):
            os.remove(os.path.join(TMP, f))
    raw = barca_raw(lam)
    atd = load_atd()
    for key, name in (('post', NAME), ('post_alt', NAME + '_reglas')):
        steps = [tuple(x) for x in e.get(key, [])]
        lab = P.V17.apply_definitions(chain(raw, ctx('barca'), steps), atd)
        out = os.path.join(OUT_DIR, 'ours_{}.csv'.format(name))
        pd.DataFrame({'frame': np.arange(len(lab)), 'label': lab}).to_csv(out, index=False)
        print('->', out, hashlib.sha256(open(out, 'rb').read()).hexdigest(), flush=True)
    json.dump(e, open(os.path.join(OUT_DIR, 'params.json'), 'w'), indent=1)
STEPS = {'final': final}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step', required=True)
    a = ap.parse_args()
    init()
    STEPS[a.step]()
if __name__ == '__main__':
    main()
