import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _find_challenge_root():
    candidates = [os.environ.get('MTAH_CHALLENGE_DIR'), os.path.join(os.path.dirname(ROOT), 'more-than-a-hack-2027'), os.path.expanduser('~/Desktop/more-than-a-hack-2027')]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, 'games.py')):
            return path
    raise IOError('no encuentro el repo oficial (more-than-a-hack-2027) al lado de este; clónalo ahí o define MTAH_CHALLENGE_DIR')
CHALLENGE_ROOT = _find_challenge_root()
if CHALLENGE_ROOT not in sys.path:
    sys.path.insert(1, CHALLENGE_ROOT)
os.environ.setdefault('MTAH_DATA_DIR', os.path.join(ROOT, 'data'))
from games import GAME, GAMES, asset_paths, half_video_blocks
CACHE_DIR = os.environ.get('MTAH_CACHE_DIR', os.path.join(ROOT, 'cache'))
GT_PUBLIC = os.path.join(CHALLENGE_ROOT, 'evaluation', 'gt', 'BAR-ATH_possession_gt_public.csv')
BASELINE_DIR = os.path.join(CHALLENGE_ROOT, 'evaluation', 'results')
FPS = GAMES[GAME]['fps']
N_FRAMES = GAMES[GAME]['n_video_frames']
PITCH_LENGTH, PITCH_WIDTH = GAMES[GAME]['pitch']
HALVES = {half: (first, last) for half, first, last in half_video_blocks(GAME)}
KICKOFF_2ND = 113742
PUBLIC_WINDOW = (11185, 18684)
HOME, AWAY = (0, 1)
LABELS = ('H', 'A', 'D', 'X')
