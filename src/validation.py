import os
import sys
import numpy as np
from src.config import CHALLENGE_ROOT, FPS, GT_PUBLIC, HALVES, KICKOFF_2ND
sys.path.insert(0, os.path.join(CHALLENGE_ROOT, 'evaluation'))
import possession_metrics as pm
K_FOLDS = 5

def _gt(gt_path=None):
    gt = pm.label_codes(pm.read_label_csv(gt_path or GT_PUBLIC))
    return (gt, pm.valid_runs(gt))

def codes(labels):
    return pm.label_codes(np.asarray(labels))

def score_public(labels):
    gt, _ = _gt()
    r = pm.evaluate(gt, codes(labels), bootstrap_n=0)
    return {'spell': r['spell_f1_mean'], 'macro': r['macro_f1']}

def cv_fixed(labels):
    gt, runs = _gt()
    spells = pm.possession_spells(gt, runs)
    pred = codes(labels)
    folds = [[i for i in range(len(spells)) if i % K_FOLDS == k] for k in range(K_FOLDS)]
    sp, ma = ([], [])
    for keep in folds:
        m = np.full(len(gt), pm.X, dtype=np.int8)
        for i in keep:
            a, b, _ = spells[i]
            m[a:b] = gt[a:b]
        r = pm.evaluate(m, pred, bootstrap_n=0)
        sp.append(r['spell_f1_mean'])
        ma.append(r['macro_f1'])
    return {'spell': float(np.mean(sp)), 'macro': float(np.mean(ma)), 'folds': sp}

def _flags(gt, runs, labels):
    filled = pm.fill_abstentions(codes(labels), runs)
    gs = pm.possession_spells(gt, runs)
    ps = pm.possession_spells(filled, runs)
    ov = pm.spell_overlaps(gs, ps)
    return (np.vstack([pm.matched_gt_flags(ov, len(gs), t) for t in pm.SPELL_IOUS]), len(ps))

def _f1(flags, n_pred, picks):
    n = len(picks)
    out = []
    for row in flags:
        tp = int(row[picks].sum())
        fp = max(0, n_pred - tp)
        out.append(2.0 * tp / (2 * tp + fp + (n - tp)) if tp else 0.0)
    return np.mean(out)

def paired_bootstrap(labels_a, labels_b, n_draws=2000, seed=0, gt_path=None):
    gt, runs = _gt(gt_path)
    fa, na = _flags(gt, runs, labels_a)
    fb, nb = _flags(gt, runs, labels_b)
    n_gt = fa.shape[1]
    allp = np.arange(n_gt)
    obs = _f1(fb, nb, allp) - _f1(fa, na, allp)
    rng = np.random.RandomState(seed)
    d = np.empty(n_draws)
    for i in range(n_draws):
        p = rng.randint(0, n_gt, n_gt)
        d[i] = _f1(fb, nb, p) - _f1(fa, na, p)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return (obs, (lo, hi), float(np.mean(d > 0)), float(np.mean(d < 0)))

def diagnostics(labels):
    labels = np.asarray(labels)
    ev = np.concatenate([labels[a:b + 1] for a, b in HALVES.values()])
    poss, dead = ([], [])
    for a, b in HALVES.values():
        seg = labels[a:b + 1]
        ch = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], ch, [len(seg)]))
        for i in range(len(bounds) - 1):
            n = (bounds[i + 1] - bounds[i]) / FPS
            (poss if seg[bounds[i]] in ('H', 'A') else dead).append(n)
    poss = np.array(poss)
    return {'H%': 100 * np.mean(ev == 'H'), 'A%': 100 * np.mean(ev == 'A'), 'D%': 100 * np.mean(ev == 'D'), 'poss': len(poss), 'mean_s': float(poss.mean()), 'lt1s': int((poss < 1).sum()), 'lt3s': int((poss < 3).sum()), 'dead_runs': len(dead), 'pre_kickoff_D%': 100 * np.mean(labels[HALVES[2][0]:KICKOFF_2ND] == 'D')}
