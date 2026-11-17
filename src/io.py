import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from src.config import CACHE_DIR, GT_PUBLIC, HOME, AWAY, N_FRAMES, PITCH_LENGTH, PITCH_WIDTH, asset_paths
N_FIELDS_PER_OBJECT = 3

def _cache_path(name):
    return os.path.join(CACHE_DIR, name)

def read_team_of_slots(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = [(t.get('id'), t.findtext('Name')) for t in root.iter('Team')]
    if len(teams) != 2:
        raise ValueError('{}: se esperaban 2 equipos, hay {}'.format(meta_path, len(teams)))
    home_id = teams[0][0]
    slots = [p.get('teamId') for p in root.iter('Player')]
    team = np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)
    return (team, (teams[0][1], teams[1][1]))

def parse_atd_raw(raw_path, n_slots):
    with open(raw_path) as handle:
        text = handle.read()
    text = text.replace(':', ',').replace(';', ',').replace('\n', ',').rstrip(',')
    values = np.array(text.split(','), dtype=np.float64)
    width = 1 + N_FIELDS_PER_OBJECT * (n_slots + 1)
    if values.size % width:
        raise ValueError('{}: {} valores no es múltiplo de {}'.format(raw_path, values.size, width))
    table = values.reshape(-1, width)
    frame_id = table[:, 0].astype(np.int64)
    players = table[:, 1:1 + N_FIELDS_PER_OBJECT * n_slots].reshape(-1, n_slots, N_FIELDS_PER_OBJECT)
    ball = table[:, 1 + N_FIELDS_PER_OBJECT * n_slots:]
    return (frame_id, ball, players)

def to_metres(xyz):
    out = xyz.astype(np.float32)
    out[..., 0] *= PITCH_LENGTH
    out[..., 1] *= PITCH_WIDTH
    return out

def build_atd_cache(source='tactical'):
    paths = asset_paths(source=source)
    team, names = read_team_of_slots(paths['atd_meta'])
    frame_id, ball, players = parse_atd_raw(paths['atd_raw'], len(team))
    if not np.array_equal(frame_id, np.arange(1, len(frame_id) + 1)):
        raise ValueError('frame_id no es 1..n contiguo')
    if source == 'tactical' and len(frame_id) != N_FRAMES:
        raise ValueError('{} registros, se esperaban {}'.format(len(frame_id), N_FRAMES))
    out = _cache_path('atd_{}.npz'.format(source))
    np.savez(out, ball=to_metres(ball), players=to_metres(players), team=team, team_names=np.array(names))
    return out

def build_tagging_cache():
    from starter.load_smart_tagging import load
    rows = []
    for event in load():
        row = {'code': event['code'].strip().upper(), 'start_s': event['start_s'], 'end_s': event['end_s'], 'start_frame': event['start_frame'], 'end_frame': event['end_frame']}
        row.update({'tag_' + k.strip().lower().replace(' ', '_'): v for k, v in event['tags'].items()})
        rows.append(row)
    table = pd.DataFrame(rows)
    out = _cache_path('smart_tagging.csv')
    table.to_csv(out, index=False)
    return out

def load_atd(source='tactical'):
    path = _cache_path('atd_{}.npz'.format(source))
    if not os.path.isfile(path):
        raise IOError('falta {} — ejecuta antes: python -m src.io'.format(path))
    with np.load(path) as data:
        return {key: data[key] for key in data.files}

def load_tagging():
    path = _cache_path('smart_tagging.csv')
    if not os.path.isfile(path):
        raise IOError('falta {} — ejecuta antes: python -m src.io'.format(path))
    return pd.read_csv(path, keep_default_na=False, na_values=[''])

def load_gt_public():
    table = pd.read_csv(GT_PUBLIC, dtype={'label': 'object'})
    return table['label'].str.strip().str.upper().values
if __name__ == '__main__':
    os.makedirs(CACHE_DIR, exist_ok=True)
    for source in ('tactical', 'broadcast'):
        print('ATD {} -> {}'.format(source, build_atd_cache(source)))
    print('Smart Tagging -> {}'.format(build_tagging_cache()))
