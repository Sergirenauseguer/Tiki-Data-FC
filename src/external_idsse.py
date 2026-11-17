import glob
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
from src.config import CACHE_DIR, ROOT
SRC_DIR = os.path.join(os.environ.get('MTAH_DATA_DIR', os.path.join(ROOT, 'data')), 'external', 'idsse')
OUT_DIR = os.path.join(CACHE_DIR, 'idsse')
KMH_TO_MS = 1 / 3.6
_FRAMESET = re.compile('<FrameSet GameSection="(\\w+)".*?TeamId="([^"]+)" PersonId="([^"]+)"')
_FRAME = re.compile('<Frame N="(\\d+)"[^>]*? X="([-\\d.]+)" Y="([-\\d.]+)"(?: Z="[-\\d.]+")? D="[-\\d.]+" S="([-\\d.]+)"(?:[^>]*?BallPossession="(\\d)" BallStatus="(\\d)")?')

def list_matches():
    return sorted((re.search('(DFL-MAT-\\w+)', f).group(1) for f in glob.glob(os.path.join(SRC_DIR, 'DFL_02_01_matchinformation_*.xml'))))

def _file(kind, match_id):
    hits = glob.glob(os.path.join(SRC_DIR, 'DFL_{}_*{}.xml'.format(kind, match_id)))
    if len(hits) != 1:
        raise IOError('falta {} de {} en {} — ejecuta python tools/get_idsse.py'.format(kind, match_id, SRC_DIR))
    return hits[0]

def match_info(match_id):
    root = ET.parse(_file('02_01_matchinformation', match_id)).getroot()
    g, env = (root.find('.//General'), root.find('.//Environment'))
    return {'home': g.get('HomeTeamId'), 'away': g.get('GuestTeamId'), 'title': g.get('MatchTitle'), 'result': g.get('Result'), 'competition': g.get('CompetitionName'), 'pitch': (float(env.get('PitchX')), float(env.get('PitchY')))}

def _parse_positions(path):
    series, cur = ({}, None)
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('<Frame '):
                m = _FRAME.match(line)
                cur[0].append(int(m.group(1)))
                cur[1].append(float(m.group(2)))
                cur[2].append(float(m.group(3)))
                cur[3].append(float(m.group(4)))
                if m.group(5) is not None:
                    cur[4].append(int(m.group(5)))
                    cur[5].append(int(m.group(6)))
            elif line.startswith('<FrameSet '):
                m = _FRAMESET.match(line)
                cur = ([], [], [], [], [], [])
                series[m.group(1), m.group(2), m.group(3)] = cur
    return series

def _half_start_times(path, ranges):
    want = {h: str(lo) for h, (lo, _, _) in ranges.items()}
    out, sec = ({}, None)
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            if line.startswith('<FrameSet ') and 'TeamId="BALL"' in line:
                sec = _FRAMESET.match(line).group(1)
            elif sec and line.startswith('<Frame N="{}"'.format(want[sec])):
                t = re.search('T="([^"]+)"', line).group(1)
                out[sec] = datetime.fromisoformat(t).timestamp()
                sec = None
                if len(out) == len(want):
                    break
    return [out['firstHalf'], out['secondHalf']]
EVENT_KINDS = ('KickOff', 'Play', 'ShotAtGoal', 'ThrowIn', 'CornerKick', 'GoalKick', 'FreeKick', 'Penalty', 'Foul', 'Offside', 'TacklingGame', 'BallClaiming', 'OtherBallAction', 'FinalWhistle', 'Substitution', 'Caution', 'SendingOff', 'RefereeBall', 'VideoAssistantAction', 'BallDeflection')

def load_events(match_id, match=None):
    import pandas as pd
    m = match if match is not None else load_match(match_id)
    info = match_info(match_id)
    team_of = {info['home']: 'H', info['away']: 'A'}
    t0, hs, n = (m['t0'], m['half_start'], len(m['label']))
    length, width = info['pitch']

    def to_frame(iso):
        if not iso:
            return np.nan
        t = datetime.fromisoformat(iso).timestamp()
        h = 1 if t >= t0[1] - 60 else 0
        f = int(round((t - t0[h]) * 25)) + int(hs[h])
        return f if 0 <= f < n else np.nan
    rows = []
    for e in ET.parse(_file('03_02_events_raw', match_id)).getroot().iter('Event'):
        top = next(iter(e), None)
        if top is None or top.tag not in EVENT_KINDS:
            continue
        inner = top.find('Play') if top.tag != 'Play' else top
        if inner is None:
            inner = top.find('ShotAtGoal')
        leaf = next(iter(inner), None) if inner is not None else next(iter(top), None)
        a = inner.attrib if inner is not None else top.attrib
        team = top.get('Team') or a.get('Team') or top.get('TeamFouler') or top.get('WinnerTeam') or ''
        x, y = (e.get('X-Position'), e.get('Y-Position'))
        rows.append({'frame': to_frame(e.get('EventTime')), 'decision_frame': to_frame(top.get('DecisionTimestamp')), 'kind': top.tag, 'sub': leaf.tag if leaf is not None else '', 'team': team_of.get(team, ''), 'x': float(x) if x else np.nan, 'y': float(y) if y else np.nan, 'height': a.get('Height', ''), 'distance': a.get('Distance', ''), 'evaluation': a.get('Evaluation', ''), 'from_open_play': a.get('FromOpenPlay', '')})
    ev = pd.DataFrame(rows).dropna(subset=['frame'])
    ev['frame'] = ev['frame'].astype(int)
    return ev.sort_values('frame').reset_index(drop=True)

def build_match(match_id):
    info = match_info(match_id)
    length, width = info['pitch']
    series = _parse_positions(_file('04_03_positions_raw_observed', match_id))
    halves = ['firstHalf', 'secondHalf']
    ball_keys = {h: k for k in series for h in halves if k[0] == h and k[1] == 'BALL'}
    ranges, half_start, offset = ({}, [], 0)
    for h in halves:
        n = np.array(series[ball_keys[h]][0])
        ranges[h] = (n.min(), n.max(), offset)
        half_start.append(offset)
        offset += n.max() - n.min() + 1
    t0 = _half_start_times(_file('04_03_positions_raw_observed', match_id), ranges)
    n_frames = offset
    people = sorted({(k[1], k[2]) for k in series if k[1] not in ('BALL', 'referee')})
    slot_of = {p: i for i, p in enumerate(people)}
    team = np.array([0 if t == info['home'] else 1 if t == info['away'] else -1 for t, _ in people], dtype=np.int8)
    if (team < 0).any():
        raise ValueError('{}: equipo desconocido en el tracking'.format(match_id))
    ball = np.full((n_frames, 3), np.nan, np.float32)
    players = np.full((n_frames, len(people), 3), np.nan, np.float32)
    label = np.full(n_frames, 'D', '<U1')
    for (h, t, p), (N, X, Y, S, BP, BS) in series.items():
        if t == 'referee':
            continue
        lo, _, off = ranges[h]
        idx = np.array(N) - lo + off
        xyz = np.column_stack([np.array(X) + length / 2, np.array(Y) + width / 2, np.array(S) * KMH_TO_MS]).astype(np.float32)
        if t == 'BALL':
            ball[idx] = xyz
            bp, bs = (np.array(BP), np.array(BS))
            label[idx] = np.where(bs == 0, 'D', np.where(bp == 1, 'H', 'A'))
        else:
            players[idx, slot_of[t, p]] = xyz
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, match_id + '.npz')
    np.savez_compressed(path, ball=ball, players=players, team=team, label=label, half_start=np.array(half_start, np.int64), t0=np.array(t0), title=np.array(info['title']), result=np.array(info['result']))
    return path

def adapt_labels(label, max_touch_s=1.0, fps=25):
    out = np.asarray(label).copy()
    starts = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1]
    ends = np.r_[starts[1:], len(out)]
    vals = out[starts]
    for i in range(1, len(starts) - 1):
        if vals[i] != 'D' and vals[i - 1] == vals[i + 1] and (vals[i - 1] != 'D') and (vals[i - 1] != vals[i]) and (ends[i] - starts[i] < max_touch_s * fps):
            out[starts[i]:ends[i]] = vals[i - 1]
    return out

def adapt_controlled(label, events, min_s=2.0, min_passes=3, fps=25):
    out = np.asarray(label).copy()
    st = np.r_[0, np.flatnonzero(out[1:] != out[:-1]) + 1]
    en = np.r_[st[1:], len(out)]
    plays = events[events.kind == 'Play']
    pf = {t: np.sort(plays.frame[plays.team == t].values) for t in ('H', 'A')}
    owner = 'D'
    for a, b in zip(st, en):
        v = out[a]
        if v == 'D':
            owner = 'D'
            continue
        if owner == 'D' or v == owner:
            owner = v
            continue
        n_pass = np.searchsorted(pf[v], b) - np.searchsorted(pf[v], a)
        if b - a >= min_s * fps or n_pass >= min_passes:
            owner = v
        else:
            out[a:b] = owner
    return out
DEGRADATION = {'version': 1, 'state_csv': os.path.join(ROOT, 'experiments', 'pipeline_approach_v19_clean', 'rules_v15.csv'), 'suffix': ''}
_MODEL = {}

def set_degradation(version, state_csv, suffix=''):
    DEGRADATION.update(version=int(version), state_csv=state_csv, suffix=suffix)
    _MODEL.clear()

def deg_tag():
    base = 'deg' if DEGRADATION['version'] == 1 else 'd{}deg'.format(DEGRADATION['version'])
    return DEGRADATION['suffix'] + base if DEGRADATION['suffix'] else base

def _degradation_model():
    if 'm' not in _MODEL:
        import pandas as pd
        from src.io import load_atd
        state = DEGRADATION['state_csv']
        if not os.path.isfile(state):
            raise IOError('falta {}: ejecuta antes el paso de reglas del pipeline'.format(state))
        rules = pd.read_csv(state, keep_default_na=False).label.values.astype('<U1')
        atd = load_atd()
        if DEGRADATION['version'] == 1:
            from src.degrade import fit_visibility
            _MODEL['m'] = fit_visibility(atd, rules)
        else:
            from src.config import HALVES, PITCH_LENGTH, PITCH_WIDTH
            from src.degrade_v2 import fit_atd
            graded = np.zeros(len(rules), bool)
            for a, b in HALVES.values():
                graded[a:b + 1] = True
            _MODEL['m'] = fit_atd(atd, rules, graded, (PITCH_LENGTH, PITCH_WIDTH))
            if DEGRADATION['version'] == 6:
                from src.degrade_v2 import fit_blackouts
                _MODEL['m']['blackout'] = fit_blackouts(atd, rules, graded)
    return _MODEL['m']

def load_degraded(match_id, seed=0):
    path = os.path.join(OUT_DIR, '{}_{}{}.npz'.format(match_id, deg_tag(), seed))
    if not os.path.isfile(path):
        if DEGRADATION['version'] == 1:
            from src.degrade import degrade
        elif DEGRADATION['version'] == 3:
            from src.degrade_v2 import degrade_v3 as degrade
        elif DEGRADATION['version'] in (4, 5):
            from functools import partial
            from src.degrade_v2 import degrade_v45
            degrade = partial(degrade_v45, smooth=DEGRADATION['version'] == 5)
        elif DEGRADATION['version'] == 6:
            from src.degrade_v2 import degrade_v6 as degrade
        else:
            from src.degrade_v2 import degrade
        d = degrade(load_match(match_id), _degradation_model(), seed)
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in d.items()})
    return load_match(match_id, path=path)

def load_match(match_id, path=None):
    path = path or os.path.join(OUT_DIR, match_id + '.npz')
    if not os.path.isfile(path):
        raise IOError('falta {} — ejecuta antes: python -m src.external_idsse'.format(path))
    with np.load(path) as data:
        out = {key: data[key] for key in data.files}
    out['title'], out['result'] = (str(out['title']), str(out['result']))
    return out
if __name__ == '__main__':
    for mid in list_matches():
        if os.path.isfile(os.path.join(OUT_DIR, mid + '.npz')):
            print('ok   {}'.format(mid))
            continue
        try:
            print('build {} -> {}'.format(mid, build_match(mid)), flush=True)
        except IOError as err:
            print('skip {}: {}'.format(mid, err))
