import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ROOT
from src.postprocess import delay_team_changes, goalkeeper_slots, keeper_hold_alive
from src.validation import cv_fixed, diagnostics, paired_bootstrap, score_public
V13_CSV = os.path.join(ROOT, 'experiments', 'pipeline_approach_v13_physics', 'ours_pipeline_approach_v13_physics.csv')
OUT_CSV = os.path.join(ROOT, 'experiments', 'pipeline_approach_v14_physics', 'ours_pipeline_approach_v14_physics.csv')
KEEPER_RADIUS_M, KEEPER_LOOK_S, KEEPER_MAX_HOLD_S = (1.5, 3.0, 15.0)
TEAM_CHANGE_DELAY_S = 0.6

def build(v13_labels, atd):
    gk = goalkeeper_slots(atd)
    out, n_keeper = keeper_hold_alive(v13_labels, atd, gk, KEEPER_RADIUS_M, KEEPER_LOOK_S, KEEPER_MAX_HOLD_S)
    out = delay_team_changes(out, TEAM_CHANGE_DELAY_S)
    return (out, gk, n_keeper)

def main():
    if not os.path.isfile(V13_CSV):
        raise IOError('falta {}: ejecuta antes `python src/pipeline_approach_v13_physics.py`'.format(V13_CSV))
    from src.io import load_atd
    v13 = pd.read_csv(V13_CSV, keep_default_na=False).label.values
    v14, gk, n_keeper = build(v13, load_atd())
    print('porteros (parte, equipo) -> slot:', gk)
    print('capa 7: {} rachas D devueltas al equipo del portero'.format(n_keeper))
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    pd.DataFrame({'frame': np.arange(len(v14)), 'label': v14}).to_csv(OUT_CSV, index=False)
    print('escrito', OUT_CSV, '({} frames cambiados respecto a v13)'.format(int((v14 != v13).sum())))
    print('\n{:<6s} {:>13s} {:>13s} {:>10s} {:>10s}'.format('', 'spell público', 'macro público', 'spell CV', 'macro CV'))
    for name, lab in (('v13', v13), ('v14', v14)):
        s, c = (score_public(lab), cv_fixed(lab))
        print('{:<6s} {:13.4f} {:13.4f} {:10.4f} {:10.4f}'.format(name, s['spell'], s['macro'], c['spell'], c['macro']))
    obs, ci, p_up, p_down = paired_bootstrap(v13, v14)
    print('bootstrap emparejado v14-v13 (ventana pública): {:+.4f} IC95 [{:+.4f}, {:+.4f}] P(mejor)={:.2f} P(peor)={:.2f}'.format(obs, ci[0], ci[1], p_up, p_down))
    print('\ndiagnóstico partido completo:')
    for name, lab in (('v13', v13), ('v14', v14)):
        d = diagnostics(lab)
        print('  {}: H {:.1f}% A {:.1f}% D {:.1f}% | {} posesiones, media {:.1f} s, <1 s: {}, <3 s: {} | rachas D {}'.format(name, d['H%'], d['A%'], d['D%'], d['poss'], d['mean_s'], d['lt1s'], d['lt3s'], d['dead_runs']))
if __name__ == '__main__':
    main()
