import numpy as np
import pandas as pd
FPS = 25
WINS = (1, 3)

def _half(pl, team, L, W, gate=0):
    x, y = (pl[:, :, 0], pl[:, :, 1])
    ok = np.isfinite(x) & np.isfinite(y)
    xs = np.where(ok, x, np.nan)
    ys = np.where(ok, y, np.nan)
    d_side = np.minimum(ys, W - ys)
    d_end = np.minimum(xs, L - xs)
    f = {}
    with np.errstate(invalid='ignore', all='ignore'):
        f['rs_n_side'] = np.nansum(d_side < 0.5, 1).astype(float)
        f['rs_dmin_side'] = np.nan_to_num(np.nanmin(d_side, 1), nan=34.0)
        f['rs_n_end'] = np.nansum(d_end < 0.5, 1).astype(float)
        dc = np.full(xs.shape, np.inf)
        for cx, cy in ((0, 0), (0, W), (L, 0), (L, W)):
            dc = np.minimum(dc, np.hypot(xs - cx, ys - cy))
        dc = np.where(ok, dc, np.inf)
        f['rs_dmin_corner'] = np.minimum(dc.min(1), 60.0)
        th, ta = (team == 0, team == 1)
        left = xs < L / 2
        nh = np.sum(ok[:, th], 1)
        na = np.sum(ok[:, ta], 1)
        sh = np.sum(left[:, th] & ok[:, th], 1) / np.maximum(nh, 1)
        sa = np.sum(left[:, ta] & ok[:, ta], 1) / np.maximum(na, 1)
        f['rs_kickoff'] = np.maximum(sh * (1 - sa), (1 - sh) * sa)
        if gate:
            f['rs_kickoff'] = np.where(np.minimum(nh, na) >= gate, f['rs_kickoff'], 0.0)
        dcen = np.where(ok, np.hypot(xs - L / 2, ys - W / 2), np.inf)
        f['rs_dmin_center'] = np.minimum(dcen.min(1), 60.0)
        v = np.full(xs.shape, np.nan)
        v[1:] = np.hypot(np.diff(xs, axis=0), np.diff(ys, axis=0)) * FPS
        v = np.where(v > 12, np.nan, v)
        f['rs_v_mean'] = np.nan_to_num(np.nanmean(v, 1), nan=0.0)
        f['rs_v_max'] = np.nan_to_num(np.nanmax(v, 1), nan=0.0)
    return f

def taker_series(atd, halves, pitch):
    L, W = pitch
    pl = np.asarray(atd['players'], float)
    team = np.asarray(atd['team'])
    sign = np.where(team == 0, 1.0, -1.0)
    out = np.zeros(len(pl), np.float32)
    for a, b in halves.values():
        x, y = (pl[a:b + 1, :, 0], pl[a:b + 1, :, 1])
        ok = np.isfinite(x) & np.isfinite(y)
        d = np.minimum(np.minimum(y, W - y), np.minimum(x, L - x))
        dc = np.full(x.shape, np.inf)
        for cx, cy in ((0, 0), (0, W), (L, 0), (L, W)):
            dc = np.minimum(dc, np.hypot(x - cx, y - cy))
        score = np.where(ok & ((d < 0.5) | (dc < 3.0)), np.minimum(d, dc - 2.5), np.inf)
        j = np.argmin(score, 1)
        has = np.isfinite(score[np.arange(len(j)), j])
        out[a:b + 1] = np.where(has, sign[j], 0.0)
    return out

def restart_features(atd, halves, pitch, gate=0):
    L, W = pitch
    pl = np.asarray(atd['players'], float)
    team = np.asarray(atd['team'])
    n = len(pl)
    out = None
    for a, b in halves.values():
        f = _half(pl[a:b + 1], team, L, W, gate)
        if out is None:
            out = {}
            for c in f:
                out[c] = np.zeros(n)
                for w in WINS:
                    out['{}_c{}'.format(c, w)] = np.zeros(n)
        for c, x in f.items():
            out[c][a:b + 1] = x
            s = pd.Series(x)
            for w in WINS:
                out['{}_c{}'.format(c, w)][a:b + 1] = s.rolling(2 * w * FPS + 1, center=True, min_periods=1).mean().values
    return pd.DataFrame({k: v.astype(np.float32) for k, v in out.items()})
