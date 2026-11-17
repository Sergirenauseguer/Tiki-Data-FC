import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ROOT
from src import physics_ball_state as BS
from src import physics_dead_ball as DB
from src.features import ball_visible
from src.postprocess import goalkeeper_slots
from src.validation import cv_fixed, diagnostics, paired_bootstrap, score_public
V14_CSV = os.path.join(ROOT, 'experiments', 'pipeline_approach_v14_physics', 'ours_pipeline_approach_v14_physics.csv')
OUT_CSV = os.path.join(ROOT, 'experiments', 'pipeline_approach_v15_physics', 'ours_pipeline_approach_v15_physics.csv')
CTRL_RADIUS_M, CTRL_REL_SPEED, CTRL_MIN_FRAMES = (1.0, 3.0, 3)
SNAP_WINDOW_S = 2.0

def build(v14_labels, atd, verbose=True):
    kin = BS.kinematics(atd)
    p90s = DB.player_p90(atd)
    oob, lost = DB.out_of_play_frames(kin)
    restarts = [f for f, _ in BS.restart_events(kin)]
    state = BS.frame_state(kin, CTRL_RADIUS_M, CTRL_REL_SPEED, CTRL_RADIUS_M)
    episodes = BS.control_episodes(state, CTRL_MIN_FRAMES)
    catches = DB.keeper_catches(kin, goalkeeper_slots(atd))
    out, n9 = DB.revive_running(v14_labels, p90s, ball_visible(atd))
    out, n10l, n10w = DB.backdate_dead_starts(out, p90s, oob, lost, restart_frames=np.asarray(restarts))
    out, n11 = DB.fill_dead_fragments(out, p90s, restarts)
    out, n12 = DB.keeper_catch_alive(out, catches)
    out, n13, _ = BS.snap_team_changes(out, episodes, SNAP_WINDOW_S)
    if verbose:
        print('capa 9  revivir D con carrera:      {} rachas'.format(n9))
        print('capa 10 inicio de parada:           {} por línea, {} por silbato'.format(n10l, n10w))
        print('capa 11 trozos dentro de paradas:   {} huecos rellenados'.format(n11))
        print('capa 12 blocajes del portero:       {} de {} detectados'.format(n12, len(catches)))
        print('capa 13 cambios al control:         {} movidos ({} controles, {} reanudaciones)'.format(n13, len(episodes), len(restarts)))
    return out
