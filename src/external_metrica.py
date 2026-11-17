import glob
import json
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from src.config import CACHE_DIR, ROOT
SRC_DIR = os.path.join(os.environ.get('MTAH_DATA_DIR', os.path.join(ROOT, 'data')), 'external', 'metrica')
OUT_DIR = os.path.join(CACHE_DIR, 'metrica')
FPS = 25
PITCH = (105.0, 68.0)
MATCHES = ('metrica_1', 'metrica_2', 'metrica_3')
EVENT_COLS = ['team', 'type', 'subtype', 'period', 'start_frame', 'end_frame', 'sx', 'sy', 'ex', 'ey', 'player', 'to']

def list_matches():
    return [m for m in MATCHES if os.path.isdir(_game_dir(m))]

def _game_dir(match_id):
    return os.path.join(SRC_DIR, 'Sample_Game_{}'.format(match_id.split('_')[1]))

def _read_csv_team(path):
    hdr = pd.read_csv(path, nrows=3, header=None)
    names = hdr.iloc[2].tolist()
    df = pd.read_csv(path, skiprows=3, header=None)
    cols = [i for i, n in enumerate(names) if isinstance(n, str) and n.startswith('Player')]
    xy = np.stack([df[cols].values.astype(np.float64), df[[c + 1 for c in cols]].values.astype(np.float64)], -1)
    ball = df.iloc[:, -2:].values.astype(np.float64)
    return (df[1].values.astype(np.int64), df[0].values.astype(np.int64), xy, [names[c] for c in cols], ball)

def _tracking_csv(match_id):
    g = match_id.split('_')[1]
    base = os.path.join(_game_dir(match_id), 'Sample_Game_{}_RawTrackingData_{{}}_Team.csv'.format(g))
    fh, ph, xyh, nh, bh = _read_csv_team(base.format('Home'))
    fa, pa, xya, na, ba = _read_csv_team(base.format('Away'))
    if not (np.array_equal(fh, fa) and np.array_equal(ph, pa)):
        raise ValueError(match_id + ': frames distintos en local y visitante')
    if not np.allclose(np.nan_to_num(bh, nan=-9), np.nan_to_num(ba, nan=-9)):
        raise ValueError(match_id + ': el balón no coincide en los dos ficheros')
    xy = np.concatenate([xyh, xya], axis=1)
    team = np.r_[np.zeros(len(nh)), np.ones(len(na))].astype(np.int8)
    return (fh, ph, xy, team, bh)

def _tracking_epts(match_id):
    d = _game_dir(match_id)
    root = ET.parse(os.path.join(d, 'Sample_Game_3_metadata.xml')).getroot()
    team_of = {p.get('id'): p.get('teamId') for p in root.iter('Player')}
    local = root.find('.//Score').get('idLocalTeam')
    chan_player = {c.get('id'): c.get('playerId') for c in root.iter('PlayerChannel')}
    halves = {}
    for prm in root.iter('ProviderParameter'):
        name, val = (prm.findtext('Name'), prm.findtext('Value'))
        if name.endswith(('_start', '_end')) and val and val.isdigit():
            halves[name] = int(val)
    specs = []
    for s in root.iter('DataFormatSpecification'):
        order = []
        for ref in s.iter('PlayerChannelRef'):
            pid = chan_player[ref.get('playerChannelId')]
            if ref.get('playerChannelId').endswith('_x'):
                order.append(pid)
        specs.append((int(s.get('startFrame')), int(s.get('endFrame')), order))
    people = sorted(set(chan_player.values()), key=lambda p: (team_of[p] != local, p))
    slot = {p: i for i, p in enumerate(people)}
    team = np.array([0 if team_of[p] == local else 1 for p in people], np.int8)
    frames, rows_xy, rows_ball = ([], [], [])
    with open(os.path.join(d, 'Sample_Game_3_tracking.txt')) as fh:
        lines = fh.read().splitlines()
    n = len(lines)
    frame = np.empty(n, np.int64)
    xy = np.full((n, len(people), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    spec_i = 0
    for i, line in enumerate(lines):
        f_str, pl, bl = line.split(':')
        f = int(f_str)
        frame[i] = f
        while not specs[spec_i][0] <= f <= specs[spec_i][1]:
            spec_i = (spec_i + 1) % len(specs)
        vals = pl.split(';')
        order = specs[spec_i][2]
        if len(vals) != len(order):
            raise ValueError('frame {}: {} jugadores, la especificación dice {}'.format(f, len(vals), len(order)))
        for pid, v in zip(order, vals):
            x, y = v.split(',')
            xy[i, slot[pid]] = (float(x), float(y))
        bx, by = bl.split(',')[:2]
        ball[i] = (float(bx), float(by))
    period = np.where(frame <= halves['first_half_end'], 1, 2)
    return (frame, period, xy, team, ball)

def read_tracking(match_id):
    if match_id == 'metrica_3':
        return _tracking_epts(match_id)
    return _tracking_csv(match_id)

def read_events(match_id):
    d = _game_dir(match_id)
    if match_id == 'metrica_3':
        raw = json.load(open(os.path.join(d, 'Sample_Game_3_events.json')))
        meta = ET.parse(os.path.join(d, 'Sample_Game_3_metadata.xml')).getroot()
        local = meta.find('.//Score').get('idLocalTeam')
        rows = []
        for e in raw['data']:
            st = e.get('subtypes')
            if isinstance(st, list):
                sub = '-'.join((s['name'] for s in st))
            else:
                sub = st['name'] if st else ''
            rows.append({'team': 'H' if e['team']['id'] == local else 'A', 'type': e['type']['name'], 'subtype': sub, 'period': e['period'], 'start_frame': e['start']['frame'], 'end_frame': e['end']['frame'], 'sx': e['start']['x'], 'sy': e['start']['y'], 'ex': e['end']['x'], 'ey': e['end']['y'], 'player': (e.get('from') or {}).get('id'), 'to': (e.get('to') or {}).get('id')})
        ev = pd.DataFrame(rows, columns=EVENT_COLS)
    else:
        g = match_id.split('_')[1]
        raw = pd.read_csv(os.path.join(d, 'Sample_Game_{}_RawEventsData.csv'.format(g)))
        ev = pd.DataFrame({'team': raw['Team'].map({'Home': 'H', 'Away': 'A'}), 'type': raw['Type'], 'subtype': raw['Subtype'].fillna(''), 'period': raw['Period'], 'start_frame': raw['Start Frame'], 'end_frame': raw['End Frame'], 'sx': raw['Start X'], 'sy': raw['Start Y'], 'ex': raw['End X'], 'ey': raw['End Y'], 'player': raw['From'], 'to': raw['To']})
    for c in ('sx', 'sy', 'ex', 'ey'):
        ev[c] = pd.to_numeric(ev[c], errors='coerce')
    ev['start_frame'] = ev['start_frame'].astype(np.int64)
    ev['end_frame'] = ev['end_frame'].astype(np.int64)
    return ev.reset_index(drop=True)
RESTART_TYPES = ('SET PIECE', 'RECOVERY', 'PASS', 'SHOT', 'CARRY')
ONSET_BEFORE_S = 1.0
ONSET_AFTER_S = 5.0
VISIBLE_MAX_S = 3.0
FILL_LIVE_S = 1.0

def _stop_start(r):
    t, s = (r.type, r.subtype)
    if t == 'BALL OUT':
        return (r.end_frame, 'fuera')
    if t == 'SHOT' and ('OUT' in s or 'GOAL' in s):
        return (r.end_frame, 'gol' if 'GOAL' in s else 'tiro fuera')
    if t == 'FAULT RECEIVED':
        return (r.start_frame, 'falta')
    if t == 'CHALLENGE' and 'FAULT' in s and ('ADVANTAGE' not in s):
        return (r.start_frame, 'falta')
    if t in ('BALL LOST', 'PASS') and 'OFFSIDE' in s:
        return (r.end_frame, 'fuera de juego')
    if t == 'BALL LOST' and ('FORCED' in s or 'HAND BALL' in s or 'END HALF' in s):
        return (r.start_frame, 'árbitro')
    return None

def _runs(mask):
    st = np.r_[0, np.flatnonzero(mask[1:] != mask[:-1]) + 1]
    en = np.r_[st[1:], len(mask)]
    return [(int(a), int(b)) for a, b in zip(st, en) if mask[a]]

def stoppages(ev, blank, first_frame=1):
    f2i = lambda f: int(f) - first_frame
    n = len(blank)
    run_start = np.full(n, -1)
    for a, b in _runs(blank):
        run_start[a:b] = a
    rows, dead = ([], None)
    for i, r in ev.iterrows():
        if dead is None:
            s = _stop_start(r)
            if s is not None:
                dead = (int(s[0]), s[1])
            continue
        if r.start_frame <= dead[0] or r.type not in RESTART_TYPES:
            continue
        rows.append({'a': dead[0], 'b': int(r.start_frame), 'causa': dead[1], 'team': r.team, 'restart': '{} {}'.format(r.type, r.subtype).strip(), 'origen': 'eventos'})
        dead = None
    st = pd.DataFrame(rows)
    keep = []
    for k, s in st.iterrows():
        ia, ib = (f2i(s.a), f2i(s.b))
        vis_all = not blank[max(ia, 0):max(ib, ia + 1)].any()
        if vis_all and s.b - s.a < VISIBLE_MAX_S * FPS:
            continue
        j = ib - 1
        if 0 <= j < n and blank[j] and (run_start[j] >= 0):
            onset = run_start[j]
            if ia - ONSET_BEFORE_S * FPS <= onset <= ia + ONSET_AFTER_S * FPS:
                st.at[k, 'a'] = onset + first_frame
                st.at[k, 'origen'] = 'eventos + tracking'
        keep.append(k)
    st = st.loc[keep].reset_index(drop=True)
    ends = set(st.b.tolist())
    extra = []
    for _, r in ev[ev.type == 'SET PIECE'].iterrows():
        if int(r.start_frame) in ends:
            continue
        j = f2i(r.start_frame) - 1
        if 0 <= j < n and blank[j] and (run_start[j] >= 0):
            extra.append({'a': run_start[j] + first_frame, 'b': int(r.start_frame), 'causa': 'sin evento', 'team': r.team, 'restart': 'SET PIECE ' + r.subtype, 'origen': 'tracking'})
    if extra:
        st = pd.concat([st, pd.DataFrame(extra)], ignore_index=True)
    st = st.sort_values('a').reset_index(drop=True)
    st['a'] = st['a'].astype(np.int64)
    st['b'] = st['b'].astype(np.int64)
    for k in range(1, len(st)):
        if st.at[k, 'a'] < st.at[k - 1, 'b']:
            st.at[k - 1, 'b'] = max(st.at[k, 'a'], st.at[k - 1, 'a'] + 1)
    return st
OWN_TYPES = ('PASS', 'CARRY', 'SHOT')
SAME_INSTANT = 3

def possession(ev, st, n, first_frame=1, drop=True):
    label = np.full(n, '', '<U1')
    dead = np.zeros(n, bool)
    for _, s in st.iterrows():
        dead[max(s.a - first_frame, 0):max(s.b - first_frame, 0)] = True
    changes = []
    for _, s in st.iterrows():
        changes.append((s.b - first_frame, s.team, 'reanudación'))
    for _, r in ev.iterrows():
        if r.type == 'RECOVERY':
            changes.append((r.start_frame - first_frame, r.team, 'RECOVERY'))
        elif r.type in OWN_TYPES or r.type == 'SET PIECE':
            changes.append((r.start_frame - first_frame, r.team, 'acción'))
    changes.sort(key=lambda c: c[0])
    owner, cur, count = ('', 0, {'reanudación': 0, 'RECOVERY': 0, 'acción (sin RECOVERY)': 0})
    implicit = []
    for f, team, src in changes:
        f = int(min(max(f, 0), n))
        label[cur:f] = owner
        cur = f
        if team != owner:
            if owner and src == 'acción':
                count['acción (sin RECOVERY)'] += 1
                implicit.append(f)
            elif owner or src == 'reanudación':
                count[src if src != 'acción' else 'acción (sin RECOVERY)'] += 1
            owner = team
    label[cur:] = owner
    label[dead] = 'D'
    label[label == ''] = 'D'
    count['filas_implicitas'] = implicit
    if not drop:
        return (label, count)
    label, count['toques sin control'] = _drop_uncontrolled(label, ev, st, first_frame)
    return (label, count)

def _drop_uncontrolled(label, ev, st, first_frame):
    out = label.copy()
    restarts = set((st.b - first_frame).astype(int).tolist())
    rec = {}
    for _, r in ev[ev.type == 'RECOVERY'].iterrows():
        rec.setdefault(int(r.start_frame - first_frame), set()).add(r.team)
    ctl = ev[ev.type.isin(OWN_TYPES + ('BALL LOST',))]
    rows = ctl.start_frame.values - first_frame
    st_ = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1]
    en_ = np.r_[st_[1:], len(out)]
    n_drop = 0
    for k in range(1, len(st_)):
        a, b, t = (st_[k], en_[k], out[st_[k]])
        prev = out[st_[k] - 1]
        if t == 'D' or prev in ('D', t) or a in restarts or (t not in rec.get(a, set())):
            continue
        sel = (rows >= a) & (rows < b) & (ctl.team.values == t)
        typ = ctl.type.values[sel]
        dur = (ctl.end_frame.values - ctl.start_frame.values)[sel]
        late_lost = (typ == 'BALL LOST') & (rows[sel] >= a + SAME_INSTANT)
        real = np.isin(typ, ('PASS', 'SHOT')) | (typ == 'CARRY') & (dur >= SAME_INSTANT)
        if real.any() or late_lost.any():
            continue
        out[a:b] = prev
        n_drop += 1
    return (out, n_drop)

def _speed(xy_m):
    v = np.full(xy_m.shape[:-1], np.nan, np.float32)
    v[1:-1] = np.hypot(*(xy_m[2:] - xy_m[:-2]).T).T * FPS / 2
    return v

def build_match(match_id):
    frame, period, xy, team, ball = read_tracking(match_id)
    ev = read_events(match_id)
    f0 = int(frame[0])
    blank = np.isnan(ball[:, 0])
    st = stoppages(ev, blank, f0)
    label_all, count = possession(ev, st, len(frame), f0)
    label_raw, _ = possession(ev, st, len(frame), f0, drop=False)
    keep, half_start = ([], [])
    for p in (1, 2):
        e = ev[ev.period == p]
        ko = e[(e.type == 'SET PIECE') & e.subtype.str.contains('KICK OFF')].start_frame.min()
        rows = np.flatnonzero(period == p)
        end = int(frame[rows[-1]]) + 1
        last = st[(st.a >= ko) & (st.a < end)]
        if len(last) and last.b.iloc[-1] >= end - 1:
            end = int(last.a.iloc[-1])
        endh = e[(e.type == 'BALL LOST') & e.subtype.str.contains('END HALF')]
        if len(endh):
            end = min(end, int(endh.start_frame.iloc[0]))
        sel = rows[(frame[rows] >= ko) & (frame[rows] < end)]
        half_start.append(sum((len(k) for k in keep)))
        keep.append(sel)
    idx = np.concatenate(keep)
    L, W = PITCH
    xy_m = (xy[idx] * np.array([L, W])).astype(np.float32)
    b_m = (ball[idx] * np.array([L, W])).astype(np.float32)
    players = np.concatenate([xy_m, np.zeros(xy_m.shape[:-1] + (1,), np.float32)], -1)
    players[..., 2] = _speed(xy_m)
    bal = np.concatenate([b_m, _speed(b_m[:, None, :])[:, 0, None]], -1)
    out = {'ball': bal, 'players': players, 'team': team, 'label': label_all[idx], 'label_raw': label_raw[idx], 'half_start': np.array(half_start, np.int64), 'src_frame': frame[idx], 'title': np.array(match_id), 'result': np.array('')}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, match_id + '.npz')
    np.savez_compressed(path, **out)
    pos = {int(f): i for i, f in enumerate(frame[idx])}
    ev = ev.assign(row=ev.start_frame.map(pos), row_end=ev.end_frame.map(pos))
    ev.to_csv(os.path.join(OUT_DIR, match_id + '_events.csv'), index=False)
    st.assign(row_a=st.a.map(pos), row_b=st.b.map(pos)).to_csv(os.path.join(OUT_DIR, match_id + '_stoppages.csv'), index=False)
    return (path, count)

def load_degraded(match_id, seed=0):
    from src import external_idsse as EI
    path = os.path.join(OUT_DIR, '{}_{}{}.npz'.format(match_id, EI.deg_tag(), seed))
    if not os.path.isfile(path):
        m = load_match(match_id)
        ball = m['ball'].copy()
        dead = m['label'] == 'D'
        vis = np.isfinite(ball[:, 0])
        for a, b in _runs(~vis & ~dead):
            if a > 0 and b < len(vis) and vis[a - 1] and vis[b] and (b - a < FILL_LIVE_S * FPS):
                w = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
                ball[a:b, :2] = ball[a - 1, :2] + w[:, None] * (ball[b, :2] - ball[a - 1, :2])
        vis = np.isfinite(ball[:, 0])
        for a, b in _runs(dead):
            if a > 0 and (not vis[a - 1]) and (not vis[a]):
                prev = np.flatnonzero(vis[max(a - 2 * FPS, 0):a])
                if len(prev):
                    ball[a] = ball[max(a - 2 * FPS, 0) + prev[-1]]
        m = dict(m, ball=ball)
        if EI.DEGRADATION['version'] == 1:
            from src.degrade import degrade
            d = degrade(m, EI._degradation_model(), seed)
        elif EI.DEGRADATION['version'] == 3:
            from src.degrade_v2 import degrade_v3
            d = degrade_v3(m, EI._degradation_model(), seed, pitch=PITCH)
        elif EI.DEGRADATION['version'] in (4, 5):
            from src.degrade_v2 import degrade_v45
            d = degrade_v45(m, EI._degradation_model(), seed, pitch=PITCH, smooth=EI.DEGRADATION['version'] == 5)
        elif EI.DEGRADATION['version'] == 6:
            from src.degrade_v2 import degrade_v6
            d = degrade_v6(m, EI._degradation_model(), seed, pitch=PITCH)
        else:
            from src.degrade_v2 import degrade
            d = degrade(m, EI._degradation_model(), seed, pitch=PITCH)
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in d.items()})
    return load_match(match_id, path=path)

def load_match(match_id, path=None):
    path = path or os.path.join(OUT_DIR, match_id + '.npz')
    if not os.path.isfile(path):
        raise IOError('falta {} — ejecuta antes: python -m src.external_metrica'.format(path))
    with np.load(path) as data:
        out = {key: data[key] for key in data.files}
    out['title'], out['result'] = (str(out['title']), str(out['result']))
    return out

def load_events(match_id):
    return pd.read_csv(os.path.join(OUT_DIR, match_id + '_events.csv'), keep_default_na=False, na_values={'sx': [''], 'sy': [''], 'ex': [''], 'ey': [''], 'row': [''], 'row_end': ['']})
if __name__ == '__main__':
    for mid in list_matches():
        path, count = build_match(mid)
        print('ok   {} -> {}  cambios de dueño: {}'.format(mid, path, count), flush=True)
