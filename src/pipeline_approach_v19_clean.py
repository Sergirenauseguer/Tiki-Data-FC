import json
import os
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C
from src import chain_generic as CG
from src import pipeline_approach_v13_physics as V13
from src import pipeline_approach_v14_physics as V14
from src import pipeline_approach_v15_physics as V15
from src import pipeline_approach_v17_learned as V17
NAME = 'pipeline_approach_v19_clean'
OUT_DIR = os.path.join(C.ROOT, 'experiments', NAME)
RULES_CSV = os.path.join(OUT_DIR, 'rules_v15.csv')
P_DEAD_RULE = 0.98
SEEDS = V17.SEEDS

def set_pieces(tagging, n):
    sp = np.zeros(n, bool)
    for e in tagging or []:
        if (e['code'] or '').strip().upper() == 'SET PIECES':
            sp[max(0, e['start_frame']):min(n, e['end_frame'])] = True
    return sp

def rules_barca(atd, tagging):
    pitch = (C.PITCH_LENGTH, C.PITCH_WIDTH)
    v6 = CG.base_v6(atd, pitch)
    v6[set_pieces(tagging, len(v6)) & np.isnan(atd['ball'][:, 0])] = 'D'
    v13, _ = V13.build(v6, tagging, atd)
    v14, _, _ = V14.build(v13, atd)
    return V15.build(v14, atd)

def step_rules():
    from src.io import load_atd
    t = time.time()
    lab = rules_barca(load_atd(), V13.load_tagging())
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame({'frame': np.arange(len(lab)), 'label': lab}).to_csv(RULES_CSV, index=False)
    print('1) reglas en el Barça → {} ({:.0f} s)'.format(RULES_CSV, time.time() - t), flush=True)
    return lab

def external_data():
    return V17.load_external(seeds=SEEDS, chain=True)

def lomo_team(data, cols):
    mids = sorted({m for m, _ in data})
    out = {}
    for mid in mids:
        t = time.time()
        models = V17.fit_models([X for (m, s), X in data.items() if m != mid], cols)
        X = data[mid, SEEDS[0]]
        _, p_home = V17.predict(models, X, cols)
        out[mid] = (X['ch_v15_d'].values > 0.5, p_home)
        print('  fuera {} ({:.0f} s)'.format(mid, time.time() - t), flush=True)
    return out

def combine(dead_rule, p_home, c_team, c_dead):
    p_dead = np.where(dead_rule, P_DEAD_RULE, 1 - P_DEAD_RULE)
    return V17.decode(p_dead, p_home, c_team, c_dead)

def tune_costs(oof, refs):
    rows = []
    for ct in V17.GRID_T:
        for cd in V17.GRID_D:
            s = [V17.score(refs[m], combine(d, ph, ct, cd)) for m, (d, ph) in oof.items()]
            rows.append({'c_team': ct, 'c_dead': cd, 'spell': float(np.mean([a for a, _ in s])), 'macro': float(np.mean([b for _, b in s])), 'por_partido': [round(float(a), 4) for a, _ in s]})
            print('  c_team {:4d} c_dead {:4d}: spell {:.3f} macro {:.3f}'.format(ct, cd, rows[-1]['spell'], rows[-1]['macro']), flush=True)
    return pd.DataFrame(rows)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rules = step_rules()
    print('2-3) externos degradados y señales', flush=True)
    data = external_data()
    cols = V17.feature_cols(next(iter(data.values())))
    print('4-5) costes del decodificador dejando un partido fuera', flush=True)
    oof = lomo_team(data, cols)
    grid = tune_costs(oof, V17.references('controlada', list(oof)))
    best = grid.sort_values(['spell', 'macro'], ascending=False).iloc[0]
    ct, cd = (float(best.c_team), float(best.c_dead))
    print('   elegido c_team {} c_dead {} (spell externos {:.3f})'.format(ct, cd, best.spell), flush=True)
    print('4) modelo de equipo con los 7 partidos → Barça', flush=True)
    import gc
    train = [X.iloc[::V17.STEP].copy() for X in data.values()]
    del data, oof
    gc.collect()
    models = V17.fit_models(train, cols, step=1)
    del train
    gc.collect()
    from src.chain_features import barca_chain
    XB = pd.concat([V17.barca_features(), barca_chain()], axis=1)
    _, p_home = V17.predict(models, XB, cols)
    from src.io import load_atd
    lab = V17.apply_definitions(combine(rules == 'D', p_home, ct, cd), load_atd())
    out_csv = os.path.join(OUT_DIR, 'ours_{}.csv'.format(NAME))
    pd.DataFrame({'frame': np.arange(len(lab)), 'label': lab}).to_csv(out_csv, index=False)
    np.savez_compressed(os.path.join(OUT_DIR, 'p_home_barca.npz'), p_home=p_home)
    grid.to_csv(os.path.join(OUT_DIR, 'grid_costes.csv'), index=False)
    with open(os.path.join(OUT_DIR, 'params.json'), 'w') as fh:
        json.dump({'c_team': ct, 'c_dead': cd, 'p_dead_rule': P_DEAD_RULE, 'ref': 'controlada', 'seeds': list(SEEDS), 'v6': {'horizon_s': CG.HORIZON_S, 'margin_m': CG.MARGIN_M, 'window': CG.WINDOW}, 'team_change_delay_s': V14.TEAM_CHANGE_DELAY_S}, fh, indent=1)
    print('-> {} ({} filas)'.format(out_csv, len(lab)))
if __name__ == '__main__':
    main()
