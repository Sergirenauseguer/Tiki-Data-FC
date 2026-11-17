import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ROOT, asset_paths
from src.postprocess import dead_by_player_speed, detect_half_kickoffs, force_pre_kickoff_dead, goal_to_kickoff_dead, prune_blips, require_control
from src.validation import cv_fixed, diagnostics, paired_bootstrap, score_public
V6_CSV = os.path.join(ROOT, 'experiments', 'pipeline_approach_v6', 'ours_pipeline_approach_v6.csv')
OUT_CSV = os.path.join(ROOT, 'experiments', 'pipeline_approach_v13_physics', 'ours_pipeline_approach_v13_physics.csv')
MAX_DEAD_S = 2.0
MAX_TEAM_S = 0.5
DEAD_WINDOW_S, DEAD_MAX_SPEED, DEAD_MIN_RUN_S = (2.0, 2.0, 3.0)
CONTROL_RADIUS_M, CONTROL_MIN_S = (1.5, 0.4)

def load_tagging():
    path = asset_paths().get('smart_tagging')
    if not path or not os.path.isfile(path):
        return None
    from starter.load_smart_tagging import load
    return load()

def load_atd_or_none():
    try:
        from src.io import load_atd
        return load_atd()
    except IOError:
        return None

def build(v6_labels, tagging, atd):
    out = v6_labels.copy()
    n_goals = None
    if tagging is not None and atd is not None:
        kickoffs = detect_half_kickoffs(tagging, atd['ball'][:, :2])
        print('saques iniciales detectados:', kickoffs)
        out = force_pre_kickoff_dead(out, kickoffs)
        out, n_goals = goal_to_kickoff_dead(out, tagging, atd['ball'][:, :2])
    out = prune_blips(out, MAX_TEAM_S, MAX_DEAD_S)
    if atd is not None:
        import warnings
        from src import features as F
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            p90 = np.nanpercentile(atd['players'][:, :, 2], 90, axis=1)
        out = dead_by_player_speed(out, p90, DEAD_WINDOW_S, DEAD_MAX_SPEED, DEAD_MIN_RUN_S)
        out = prune_blips(out, MAX_TEAM_S, MAX_DEAD_S)
        d_home, d_away, _, _ = F.nearest_by_team(atd)
        out = require_control(out, d_home, d_away, F.ball_visible(atd), CONTROL_RADIUS_M, CONTROL_MIN_S)
    return (out, n_goals)

def main():
    if not os.path.isfile(V6_CSV):
        raise IOError('falta {}: ejecuta antes `python src/pipeline_approach_v6_deprecated.py`'.format(V6_CSV))
    v6 = pd.read_csv(V6_CSV, keep_default_na=False).label.values
    tagging = load_tagging()
    atd = load_atd_or_none()
    if atd is None:
        print('capas 5-6 y gol->saque: SALTADAS (falta la caché: python -m src.io)')
    v13, n_goals = build(v6, tagging, atd)
    print('capa gol->saque:', 'SALTADA (falta Smart Tagging o la caché)' if n_goals is None else '; '.join(('gol {}: {}'.format(i + 1, c) for i, c in enumerate(n_goals))))
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame({'frame': np.arange(len(v13)), 'label': v13}).to_csv(OUT_CSV, index=False)
    print('escrito', OUT_CSV, '({} frames cambiados respecto a v6)'.format(int((v13 != v6).sum())))
    print('\n{:<6s} {:>13s} {:>13s} {:>10s} {:>10s}'.format('', 'spell público', 'macro público', 'spell CV', 'macro CV'))
    for name, lab in (('v6', v6), ('v13', v13)):
        s, c = (score_public(lab), cv_fixed(lab))
        print('{:<6s} {:13.4f} {:13.4f} {:10.4f} {:10.4f}'.format(name, s['spell'], s['macro'], c['spell'], c['macro']))
    obs, ci, p_up, p_down = paired_bootstrap(v6, v13)
    print('bootstrap emparejado v13-v6 (ventana pública): {:+.4f} IC95 [{:+.4f}, {:+.4f}] P(mejor)={:.2f} P(peor)={:.2f}'.format(obs, ci[0], ci[1], p_up, p_down))
    print('\ndiagnóstico partido completo:')
    for name, lab in (('v6', v6), ('v13', v13)):
        d = diagnostics(lab)
        print('  {}: H {:.1f}% A {:.1f}% D {:.1f}% | {} posesiones, media {:.1f} s, <1 s: {}, <3 s: {} | rachas D {} | D antes del saque 2ª: {:.0f}%'.format(name, d['H%'], d['A%'], d['D%'], d['poss'], d['mean_s'], d['lt1s'], d['lt3s'], d['dead_runs'], d['pre_kickoff_D%']))
if __name__ == '__main__':
    main()
