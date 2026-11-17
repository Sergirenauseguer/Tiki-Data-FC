import warnings
import numpy as np
import pandas as pd
FPS = 25
PROJ_S = 0.4

def _cs(v):
    return np.concatenate(([0.0], np.cumsum(v)))

def _win_mean(v, lo_s, hi_s, fps=FPS):
    v = np.asarray(v, float)
    ok = ~np.isnan(v)
    s, c = (_cs(np.where(ok, v, 0.0)), _cs(ok.astype(float)))
    n = len(v)
    i = np.arange(n)
    lo = np.clip(i + int(round(lo_s * fps)), 0, n)
    hi = np.clip(i + int(round(hi_s * fps)) + 1, 0, n)
    cnt = c[hi] - c[lo]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cnt > 0, (s[hi] - s[lo]) / np.maximum(cnt, 1), np.nan)

def _win_max(v, lo_s, hi_s, fps=FPS):
    s = pd.Series(np.asarray(v, float))
    w = int(round((hi_s - lo_s) * fps)) + 1
    shift = int(round(hi_s * fps))
    return s.rolling(w, min_periods=1).max().shift(-shift).values

def _central_velocity(xy, k=2, fps=FPS):
    v = np.full_like(xy, np.nan, dtype=float)
    v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (fps / (2.0 * k))
    return v

def _gap_features(visible):
    n = len(visible)
    idx = np.arange(n)
    last = np.where(visible, idx, -1)
    last = np.maximum.accumulate(last)
    nxt = np.where(visible, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    since = np.where(last >= 0, (idx - last) / FPS, 60.0)
    until = np.where(nxt < n, (nxt - idx) / FPS, 60.0)
    return (np.minimum(since, 60.0), np.minimum(until, 60.0), last, np.where(nxt < n, nxt, -1))

def attack_sign(atd, halves):
    P, team = (atd['players'][:, :, 0], atd['team'])
    n = len(P)
    sign = np.ones(n)
    bounds = sorted(halves.values())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for k, (a, b) in enumerate(bounds):
            xh = np.nanmean(P[a:b + 1][:, team == 0])
            xa = np.nanmean(P[a:b + 1][:, team == 1])
            s = 1.0 if xh < xa else -1.0
            lo = 0 if k == 0 else (bounds[k - 1][1] + a) // 2 + 1
            hi = n if k == len(bounds) - 1 else (b + bounds[k + 1][0]) // 2 + 1
            sign[lo:hi] = s
    return sign

def build_features(atd, halves, pitch):
    L, W = pitch
    ball = atd['ball'][:, :2].astype(float)
    P = atd['players'][:, :, :2].astype(float)
    spd = atd['players'][:, :, 2].astype(float)
    team = np.asarray(atd['team'])
    n = len(ball)
    f = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        vis = ~np.isnan(ball[:, 0])
        bv = _central_velocity(ball)
        bspeed = np.hypot(bv[:, 0], bv[:, 1])
        bx, by = (ball[:, 0], ball[:, 1])
        d_side = np.minimum(by, W - by)
        d_end = np.minimum(bx, L - bx)
        f['ball_vis'] = vis.astype(float)
        f['ball_speed'] = np.minimum(bspeed, 40.0)
        f['ball_out'] = np.where(vis, ((bx < 0) | (bx > L) | (by < 0) | (by > W)).astype(float), np.nan)
        f['ball_d_side'] = d_side
        f['ball_d_end'] = d_end
        f['ball_dc'] = np.hypot(bx - L / 2, by - W / 2)
        f['ball_ax'] = np.abs(bx - L / 2) / (L / 2)
        for lo, hi, tag in ((-1, 1, 'c1'), (-3, 3, 'c3'), (-8, 8, 'c8'), (-3, 0, 'p3'), (0, 3, 'f3')):
            f['vis_' + tag] = _win_mean(vis.astype(float), lo, hi)
        for lo, hi, tag in ((-1, 0, 'p1'), (0, 1, 'f1'), (-3, 0, 'p3'), (0, 3, 'f3')):
            f['bspeed_' + tag] = _win_mean(np.minimum(bspeed, 40.0), lo, hi)
        f['bspeed_max_f2'] = _win_max(np.nan_to_num(np.minimum(bspeed, 40.0)), 0, 2)
        f['bspeed_max_p2'] = _win_max(np.nan_to_num(np.minimum(bspeed, 40.0)), -2, 0)
        disp = np.full(n, np.nan)
        k = 2 * FPS
        disp[k:] = np.hypot(bx[k:] - bx[:-k], by[k:] - by[:-k])
        f['ball_disp_p2'] = disp
        f['ball_disp_f2'] = np.r_[disp[k:], np.full(k, np.nan)]
        since, until, last, nxt = _gap_features(vis)
        bs = pd.Series(np.minimum(bspeed, 40.0))
        bs_ff, bs_bf = (bs.ffill().values, bs.bfill().values)
        f['gap_since'], f['gap_until'] = (since, until)
        f['gap_len'] = np.where(vis, 0.0, np.minimum(since + until, 120.0))
        f['gap_frac'] = np.where(vis, np.nan, since / np.maximum(since + until, 1e-06))
        for name, ref in (('last', last), ('next', nxt)):
            okr = ref >= 0
            r = np.where(okr, ref, 0)
            f['gap_%s_d_side' % name] = np.where(okr & ~vis, d_side[r], np.nan)
            f['gap_%s_d_end' % name] = np.where(okr & ~vis, d_end[r], np.nan)
            edge = bs_ff if name == 'last' else bs_bf
            f['gap_%s_speed' % name] = np.where(okr & ~vis, edge[r], np.nan)
            f['gap_%s_ax' % name] = np.where(okr & ~vis, f['ball_ax'][r], np.nan)
        f['gap_move'] = np.where(~vis & (last >= 0) & (nxt >= 0), np.hypot(bx[np.maximum(nxt, 0)] - bx[np.maximum(last, 0)], by[np.maximum(nxt, 0)] - by[np.maximum(last, 0)]), np.nan)
        pres = ~np.isnan(P[:, :, 0])
        f['n_players'] = pres.sum(axis=1).astype(float)
        p90 = np.nanpercentile(spd, 90, axis=1)
        p50 = np.nanpercentile(spd, 50, axis=1)
        pmax = np.nanmax(spd, axis=1)
        f['p90'], f['p50'], f['pmax'] = (p90, p50, pmax)
        for lo, hi, tag in ((-1, 1, 'c1'), (-3, 3, 'c3'), (-8, 8, 'c8'), (-3, 0, 'p3'), (0, 3, 'f3'), (-6, -3, 'p6'), (3, 6, 'f6'), (-1, 0, 'p1'), (0, 1, 'f1')):
            f['p90_' + tag] = _win_mean(p90, lo, hi)
        f['p50_c3'] = _win_mean(p50, -3, 3)
        f['p90_dfp'] = f['p90_f3'] - f['p90_p3']
        f['p90_min_c5'] = -_win_max(-np.nan_to_num(p90, nan=0.0), -5, 5)
        f['n_fast'] = np.sum(spd > 4.0, axis=1).astype(float)
        f['n_fast_c2'] = _win_mean(f['n_fast'], -2, 2)
        sp_h = np.nanmean(np.where(team[None] == 0, spd, np.nan), axis=1)
        sp_a = np.nanmean(np.where(team[None] == 1, spd, np.nan), axis=1)
        f['spd_h'], f['spd_a'] = (_win_mean(sp_h, -1, 1), _win_mean(sp_a, -1, 1))
        sgn = attack_sign(atd, halves)
        for t, s in ((0, 'h'), (1, 'a')):
            m = team[None] == t
            X = np.where(m, P[:, :, 0], np.nan)
            Y = np.where(m, P[:, :, 1], np.nan)
            f['width_' + s] = _win_mean(np.nanstd(Y, axis=1), -1, 1)
            f['length_' + s] = _win_mean(np.nanstd(X, axis=1), -1, 1)
            cx = np.nanmean(X, axis=1)
            dirn = sgn if t == 0 else -sgn
            f['cx_' + s] = _win_mean((cx - L / 2) * dirn / (L / 2), -1, 1)
            f['ball_att_' + s] = (bx - L / 2) * dirn / (L / 2)
        f['block_gap'] = np.abs(f['cx_h'] + f['cx_a'])
        in_box = ((P[:, :, 0] < 16.5) | (P[:, :, 0] > L - 16.5)) & (np.abs(P[:, :, 1] - W / 2) < 20.16)
        f['n_in_box'] = _win_mean(in_box.sum(axis=1).astype(float), -1, 1)
        near_side = (P[:, :, 1] < 3) | (P[:, :, 1] > W - 3)
        f['n_near_side'] = near_side.sum(axis=1).astype(float)
        out_p = (P[:, :, 0] < -0.5) | (P[:, :, 0] > L + 0.5) | (P[:, :, 1] < -0.5) | (P[:, :, 1] > W + 0.5)
        f['n_out_players'] = _win_mean(out_p.sum(axis=1).astype(float), -1, 1)
        dist = np.linalg.norm(P - ball[:, None, :], axis=2)
        pv = _central_velocity(P.reshape(n, -1)).reshape(P.shape)
        proj = np.linalg.norm(P + pv * PROJ_S - (ball + bv * PROJ_S)[:, None, :], axis=2)
        proj = np.where(np.isnan(proj), dist, proj)
        relv = np.linalg.norm(pv - bv[:, None, :], axis=2)
        near = {}
        for t, s in ((0, 'h'), (1, 'a')):
            m = team[None] == t
            d = np.where(m, dist, np.nan)
            ds = np.sort(np.where(np.isnan(d), np.inf, d), axis=1)
            d1 = np.where(np.isinf(ds[:, 0]), np.nan, ds[:, 0])
            d2 = np.where(np.isinf(ds[:, 1]), np.nan, ds[:, 1])
            near[s] = d1
            f['d1_' + s] = np.minimum(d1, 60.0)
            f['d2_' + s] = np.minimum(d2, 60.0)
            f['dp_' + s] = np.minimum(np.nanmin(np.where(m, proj, np.nan), axis=1), 60.0)
            j = np.nanargmin(np.where(m & ~np.isnan(dist), dist, np.inf), axis=1)
            f['relv_' + s] = np.minimum(relv[np.arange(n), j], 30.0)
            f['n3_' + s] = np.sum(m & (dist < 3.0), axis=1).astype(float)
            f['n10_' + s] = np.sum(m & (dist < 10.0), axis=1).astype(float)
            f['d1_%s_c1' % s] = _win_mean(np.minimum(d1, 60.0), -1, 1)
        f['d1_ha'] = f['d1_h'] - f['d1_a']
        f['dp_ha'] = f['dp_h'] - f['dp_a']
        f['n10_all'] = f['n10_h'] + f['n10_a']
        f['d1_min'] = np.fmin(f['d1_h'], f['d1_a'])
        f['d1_min_c1'] = _win_mean(f['d1_min'], -1, 1)
        vote = np.where(np.isnan(f['dp_ha']), np.nan, np.sign(-f['dp_ha']))
        for lo, hi, tag in ((-0.5, 0.5, 'c05'), (-2, 2, 'c2'), (-5, 5, 'c5'), (-10, 10, 'c10'), (-3, 0, 'p3'), (0, 3, 'f3'), (-8, 0, 'p8'), (0, 8, 'f8')):
            f['vote_' + tag] = _win_mean(vote, lo, hi)
        close = ~np.isnan(f['d1_min']) & (f['d1_min'] < 2.0)
        own = np.where(close, np.sign(-f['d1_ha']), np.nan)
        f['last_close'], f['since_close'] = _ffill_with_age(own)
        f['next_close'], f['until_close'] = _ffill_with_age(own[::-1])
        f['next_close'], f['until_close'] = (f['next_close'][::-1], f['until_close'][::-1])
        ctrl = _control_sign(atd, pitch)
        f['last_ctrl'], f['since_ctrl'] = _ffill_with_age(ctrl)
        nc, uc = _ffill_with_age(ctrl[::-1])
        f['next_ctrl'], f['until_ctrl'] = (nc[::-1], uc[::-1])
        f['ctrl_c3'] = _win_mean(ctrl, -3, 3)
        f['ctrl_p5'] = _win_mean(ctrl, -5, 0)
        f['ctrl_f5'] = _win_mean(ctrl, 0, 5)
    out = pd.DataFrame(f).astype(np.float32)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

def _ffill_with_age(v):
    v = np.asarray(v, float)
    n = len(v)
    idx = np.where(~np.isnan(v), np.arange(n), -1)
    idx = np.maximum.accumulate(idx)
    val = np.where(idx >= 0, v[np.maximum(idx, 0)], np.nan)
    age = np.where(idx >= 0, (np.arange(n) - idx) / FPS, 60.0)
    return (val, np.minimum(age, 60.0))

def _control_sign(atd, pitch):
    from src import physics_ball_state as B
    kin = B.kinematics(atd)
    st = B.frame_state(kin)
    out = np.full(len(atd['ball']), np.nan)
    for e in B.control_episodes(st):
        a, b, t = (e[0], e[1], e[2])
        out[a:b] = 1.0 if t == 0 else -1.0
    return out

def swap_teams(X):
    Y = X.copy()
    cols = set(X.columns)
    for c in X.columns:
        if c.endswith('_h') and c[:-2] + '_a' in cols:
            Y[c], Y[c[:-2] + '_a'] = (X[c[:-2] + '_a'].values, X[c].values)
        for mid in ('_h_', '_a_'):
            if mid in c:
                other = c.replace(mid, '_a_' if mid == '_h_' else '_h_')
                if other in cols:
                    Y[c] = X[other].values
    for c in X.columns:
        if c.endswith('_ha') or c.startswith(('vote_', 'last_close', 'next_close', 'last_ctrl', 'next_ctrl', 'ctrl_')):
            Y[c] = -X[c].values
    return Y
