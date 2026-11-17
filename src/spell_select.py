import numpy as np
from src import spell_decoding as SD
FPS = 25
IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)

def spells(lab):
    st, en = SD.runs(lab)
    v = np.asarray(lab)[st]
    k = (v == 'H') | (v == 'A')
    return (st[k], en[k], v[k])

def change_peaks(pch, thr, sep=12):
    p = np.convolve(np.asarray(pch, float), np.ones(5) / 5, mode='same')
    cand = np.flatnonzero((p >= thr) & (p >= np.r_[0, p[:-1]]) & (p >= np.r_[p[1:], 0]))
    out = []
    for c in cand[np.argsort(-p[cand], kind='stable')]:
        if all((abs(c - o) >= sep for o in out)):
            out.append(int(c))
    return np.sort(np.array(out, int))

def insert_short(lab, pch, thr, lo=0.7, hi=6.0, edge=6):
    out = np.asarray(lab).copy()
    pk = change_peaks(pch, thr)
    st, en, v = spells(out)
    for a, b, t in zip(st, en, v):
        q = pk[(pk > a + edge) & (pk < b - edge)]
        i = 0
        while i < len(q) - 1:
            if lo * FPS <= q[i + 1] - q[i] <= hi * FPS:
                out[q[i]:q[i + 1]] = 'A' if t == 'H' else 'H'
                i += 2
            else:
                i += 1
    return out

def candidates(labs):
    seen = {}
    for name, lab in labs.items():
        s, e, v = spells(lab)
        for a, b, t in zip(s, e, v):
            k = (int(a), int(b), t)
            seen.setdefault(k, []).append(name)
    keys = sorted(seen, key=lambda k: (k[0], k[1]))
    S = np.array([k[0] for k in keys])
    E = np.array([k[1] for k in keys])
    T = np.array([k[2] for k in keys])
    n = np.array([len(seen[k]) for k in keys])
    return (S, E, T, n, [seen[k] for k in keys])

def add_shifts(S, E, T, cnt, origin, n, deltas=(10, 25)):
    have = set(zip(S.tolist(), E.tolist(), T.tolist()))
    add = []
    for a, b, t in zip(S.tolist(), E.tolist(), T.tolist()):
        for d in deltas:
            for a2, b2 in ((a - d, b + d), (a + d, b - d)):
                a2, b2 = (max(a2, 0), min(b2, n))
                if b2 - a2 >= 12 and (a2, b2, t) not in have:
                    have.add((a2, b2, t))
                    add.append((a2, b2, t))
    if not add:
        return (S, E, T, cnt, origin)
    A = np.array(add, dtype=object)
    S2 = np.r_[S, A[:, 0].astype(int)]
    E2 = np.r_[E, A[:, 1].astype(int)]
    T2 = np.r_[T, A[:, 2].astype(T.dtype)]
    o = np.argsort(S2, kind='stable')
    origin2 = list(origin) + [['shift']] * len(add)
    cnt2 = np.r_[cnt, np.zeros(len(add), int)]
    return (S2[o], E2[o], T2[o], cnt2[o], [origin2[i] for i in o])

def best_iou(S, E, T, ref):
    gs, ge, gv = spells(ref)
    out = np.zeros(len(S))
    j0 = np.searchsorted(ge, S, side='right')
    for i in range(len(S)):
        j = j0[i]
        while j < len(gs) and gs[j] < E[i]:
            if gv[j] == T[i]:
                inter = min(E[i], ge[j]) - max(S[i], gs[j])
                union = max(E[i], ge[j]) - min(S[i], gs[j])
                out[i] = max(out[i], inter / union)
            j += 1
    return out

def match_value(iou):
    return np.mean([iou >= t for t in IOUS], axis=0)

def select(S, E, w, gap=0):
    order = np.argsort(E, kind='stable')
    S, E, w = (S[order], E[order], w[order])
    n = len(S)
    best = np.zeros(n + 1)
    take = np.zeros(n, bool)
    prev = np.searchsorted(E, S - gap, side='right')
    for i in range(n):
        inc = w[i] + best[prev[i]]
        if inc > best[i]:
            best[i + 1], take[i] = (inc, True)
        else:
            best[i + 1] = best[i]
    sel, i = ([], n)
    while i > 0:
        if take[i - 1]:
            sel.append(i - 1)
            i = prev[i - 1]
        else:
            i -= 1
    return order[np.array(sel[::-1], dtype=int)] if sel else np.array([], int)

def render(n, S, E, T, idx, fill='D'):
    lab = np.full(n, fill, dtype='<U1')
    for i in sorted(idx, key=lambda i: S[i]):
        lab[S[i]:E[i]] = T[i]
    for i in sorted(idx, key=lambda i: S[i]):
        if S[i] > 0 and lab[S[i] - 1] == T[i]:
            lab[S[i]] = 'D'
    return lab

def _win_mean(x, a, b):
    a, b = (max(a, 0), min(b, len(x)))
    return float(np.mean(x[a:b])) if b > a else np.nan

def _win_max(x, a, b):
    a, b = (max(a, 0), min(b, len(x)))
    return float(np.max(x[a:b])) if b > a else np.nan

def features(S, E, T, cnt, origin, labs, ph, pdead, pch, vis, ctrl_team, kick, order, extra=False):
    import pandas as pd
    sp = {k: spells(v) for k, v in labs.items()}
    rank = {k: i for i, k in enumerate(order)}
    cs_ph = np.r_[0, np.cumsum(ph)]
    rows = []
    w1, w2 = (FPS, 2 * FPS)
    for i in range(len(S)):
        a, b, t = (int(S[i]), int(E[i]), T[i])
        sg = 1.0 if t == 'H' else -1.0
        pt = ph[a:b] if t == 'H' else 1 - ph[a:b]
        L = b - a
        before = _win_mean(ph, a - w2, a)
        after = _win_mean(ph, b, b + w2)
        if t == 'A':
            before, after = (1 - before if before == before else before, 1 - after if after == after else after)
        ct = ctrl_team[a:b]
        stab = 0
        for k, (s2, e2, v2) in sp.items():
            j = np.searchsorted(e2, a, side='right')
            while j < len(s2) and s2[j] < b:
                if v2[j] == t:
                    inter = min(b, e2[j]) - max(a, s2[j])
                    if inter / (max(b, e2[j]) - min(a, s2[j])) >= 0.8:
                        stab += 1
                        break
                j += 1
        rk = [rank.get(o, len(order)) for o in origin[i]]
        rows.append({'dur': L / FPS, 'logdur': np.log(L / FPS + 0.001), 'pt_mean': float(pt.mean()), 'pt_min': float(pt.min()), 'pt_p10': float(np.quantile(pt, 0.1)), 'pt_p50': float(np.median(pt)), 'pt_first': float(pt[:w1].mean()), 'pt_last': float(pt[-w1:].mean()), 'pt_before': before, 'pt_after': after, 'pd_mean': float(pdead[a:b].mean()), 'pd_before': _win_mean(pdead, a - w1, a), 'pd_after': _win_mean(pdead, b, b + w1), 'pch_start': _win_max(pch, a - 12, a + 13), 'pch_end': _win_max(pch, b - 12, b + 13), 'pch_mean': float(pch[a:b].mean()), 'vis': float(vis[a:b].mean()), 'vis_start': _win_mean(vis, a - 12, a + 13), 'vis_end': _win_mean(vis, b - 12, b + 13), 'ctrl_own': float(np.sum(ct == sg)) / FPS, 'ctrl_opp': float(np.sum(ct == -sg)) / FPS, 'kick_start': _win_max(kick, a - w1, a + w1), 'n_dec': len(origin[i]), 'stab': stab, 'rk_min': min(rk), 'rk_max': max(rk)})
    F = pd.DataFrame(rows)
    if extra:
        F = pd.concat([F, extra_features(S, E, T, labs, ph, pdead, order)], axis=1)
    if extra and int(extra) == 2:
        F = pd.concat([F, split_features(S, E, T, ph, pch, ctrl_team)], axis=1)
    return F

def split_features(S, E, T, ph, pch, ctrl_team):
    import pandas as pd
    n = len(S)
    out = {k: np.zeros(n) for k in ('pch_max_in', 'pch_peaks_in', 'low_run', 'low_frac', 'opp_ctrl_run', 'ctrl_switch')}
    for i in range(n):
        a, b, t = (int(S[i]), int(E[i]), T[i])
        sg = 1.0 if t == 'H' else -1.0
        k = min(12, (b - a) // 4)
        inner = pch[a + k:b - k] if b - a > 2 * k + 1 else pch[a:b]
        out['pch_max_in'][i] = inner.max() if len(inner) else 0.0
        out['pch_peaks_in'][i] = float(np.sum((inner[1:-1] > 0.5) & (inner[1:-1] >= inner[:-2]) & (inner[1:-1] > inner[2:]))) if len(inner) > 2 else 0.0
        pt = ph[a:b] if t == 'H' else 1 - ph[a:b]
        low = pt < 0.5
        out['low_frac'][i] = low.mean()
        if low.any():
            d = np.diff(np.r_[0, low.astype(int), 0])
            out['low_run'][i] = (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max() / FPS
        c = ctrl_team[a:b]
        opp = c == -sg
        if opp.any():
            d = np.diff(np.r_[0, opp.astype(int), 0])
            out['opp_ctrl_run'][i] = (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max() / FPS
        nz = c[c != 0]
        out['ctrl_switch'][i] = float(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0.0
    return pd.DataFrame(out)
SIGNED = ['vote_c2', 'vote_c5', 'vote_p3', 'vote_f3', 'ctrl_c3', 'ctrl_p5', 'ctrl_f5', 'd1_ha', 'dp_ha', 'ch_v15_ha', 'last_ctrl', 'last_close']
UNSIGNED = ['ball_vis', 'ball_speed', 'n_fast', 'p90', 'n_in_box', 'gap_frac']
EDGE = ['vote_c2', 'ctrl_c3', 'ch_v15_ha']

def frame_features(S, E, T, F):
    import pandas as pd
    sg = np.where(T == 'H', 1.0, -1.0)
    L = (E - S).astype(float)
    out = {}
    for c in SIGNED + UNSIGNED:
        x = np.nan_to_num(F[c].values.astype(float)) if c in F else np.zeros(len(F))
        cs = np.r_[0.0, np.cumsum(x)]
        m = (cs[E] - cs[S]) / np.maximum(L, 1)
        out['fm_' + c] = m * sg if c in SIGNED else m
        if c in EDGE:
            k = np.minimum(FPS, np.maximum(L // 4, 1)).astype(int)
            out['fs_' + c] = (cs[S + k] - cs[S]) / k * sg
            out['fe_' + c] = (cs[E] - cs[E - k]) / k * sg
    return pd.DataFrame(out)

def taker_features(S, E, T, tk):
    import pandas as pd
    sg = np.where(T == 'H', 1.0, -1.0)
    cs = np.r_[0.0, np.cumsum(tk.astype(float))]
    cn = np.r_[0.0, np.cumsum((tk != 0).astype(float))]
    n = len(tk)

    def win(a, b):
        a, b = (np.clip(a, 0, n), np.clip(b, 0, n))
        k = np.maximum(b - a, 1)
        return ((cs[b] - cs[a]) / k, (cn[b] - cn[a]) / k)
    m0, f0 = win(S - 3 * FPS, S + FPS // 2)
    m1, f1 = win(E - FPS // 2, E + 3 * FPS)
    mi, fi = win(S, E)
    return pd.DataFrame({'tk_start': m0 * sg, 'tk_start_any': f0, 'tk_end': m1 * sg, 'tk_end_any': f1, 'tk_in': mi * sg, 'tk_in_any': fi})
RS_EDGE = ['rs_n_side_c1', 'rs_n_end_c1', 'rs_dmin_corner_c1', 'rs_kickoff_c1', 'rs_v_mean_c3']

def restart_edge_features(S, E, R):
    import pandas as pd
    n = len(R)
    out = {}
    for c in RS_EDGE:
        x = np.nan_to_num(R[c].values.astype(float))
        cs = np.r_[0.0, np.cumsum(x)]

        def win(a, b):
            a, b = (np.clip(a, 0, n), np.clip(b, 0, n))
            return (cs[b] - cs[a]) / np.maximum(b - a, 1)
        out['rse_s_' + c] = win(S - 2 * FPS, S)
        out['rse_e_' + c] = win(E, E + 2 * FPS)
    return pd.DataFrame(out)

def context_features(S, E, T, p, idx):
    import pandas as pd
    n = len(S)
    sel = np.zeros(n, bool)
    sel[idx] = True
    o = np.argsort(S[idx]) if len(idx) else np.array([], int)
    sS, sE, sT = (S[idx][o], E[idx][o], T[idx][o])
    out = {k: np.zeros(n) for k in ('is_sel', 'p1', 'iou_sel_same', 'cov_sel_other', 'gap_prev', 'gap_next', 'prev_same', 'next_same')}
    out['is_sel'] = sel.astype(float)
    out['p1'] = p
    for i in range(n):
        a, b, t = (S[i], E[i], T[i])
        j0 = np.searchsorted(sS, b, side='left')
        best, cov = (0.0, 0)
        j = j0 - 1
        while j >= 0 and sE[j] > a - 2000:
            if sE[j] > a:
                inter = min(b, sE[j]) - max(a, sS[j])
                if inter > 0:
                    if sT[j] == t:
                        best = max(best, inter / (max(b, sE[j]) - min(a, sS[j])))
                    else:
                        cov += inter
            j -= 1
        out['iou_sel_same'][i] = best
        out['cov_sel_other'][i] = cov / (b - a)
        k = np.searchsorted(sE, a, side='right') - 1
        if k >= 0:
            out['gap_prev'][i] = (a - sE[k]) / FPS
            out['prev_same'][i] = float(sT[k] == t)
        else:
            out['gap_prev'][i] = 99.0
        k2 = np.searchsorted(sS, b, side='left')
        if k2 < len(sS):
            out['gap_next'][i] = (sS[k2] - b) / FPS
            out['next_same'][i] = float(sT[k2] == t)
        else:
            out['gap_next'][i] = 99.0
    return pd.DataFrame(out)

def extra_features(S, E, T, labs, ph, pdead, order):
    import pandas as pd
    mid = labs[order[len(order) // 2]]
    n = len(S)
    agree, edge_in, edge_out, pdmax, rival = (np.zeros(n) for _ in range(5))
    o_s = np.argsort(S)
    Ss, Es, Ts = (S[o_s], E[o_s], T[o_s])
    for i in range(n):
        a, b, t = (int(S[i]), int(E[i]), T[i])
        agree[i] = np.mean(mid[a:b] == t)
        pt = ph if t == 'H' else 1 - ph
        k = min(12, max((b - a) // 4, 1))
        edge_in[i] = 0.5 * (pt[a:a + k].mean() + pt[b - k:b].mean())
        lo, hi = (max(a - k, 0), min(b + k, len(ph)))
        o = [x for x in (pt[lo:a], pt[b:hi]) if len(x)]
        edge_out[i] = np.mean([x.mean() for x in o]) if o else 0.5
        pdmax[i] = pdead[a:b].max()
        j0 = np.searchsorted(Ss, b, side='left')
        best, j = (0.0, j0 - 1)
        while j >= 0 and j0 - j < 60:
            if Ts[j] != t and Es[j] > a:
                best = max(best, (min(b, Es[j]) - max(a, Ss[j])) / (b - a))
            j -= 1
        rival[i] = best
    return pd.DataFrame({'agree_mid': agree, 'edge_in': edge_in, 'edge_out': edge_out, 'edge_diff': edge_in - edge_out, 'pd_max': pdmax, 'rival_cov': rival})
