import os
import sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)
import pipeline_approach_v21 as P
OUT = os.path.join(ROOT, 'cache', 'seq')
STEP = 5

def main():
    P.setup(5)
    os.makedirs(OUT, exist_ok=True)
    cols = None
    for m in P.EI.list_matches():
        for s in (0, 1):
            X = P.frames(m, s)
            if cols is None:
                cols = P.V17.feature_cols(X)
            lab = X['label'].values[::STEP]
            Xd = X[cols].iloc[::STEP].reset_index(drop=True)
            Xs = P.V17.swap_teams(Xd)
            np.savez_compressed(os.path.join(OUT, '{}_{}.npz'.format(m, s)), X=Xd.values.astype(np.float32), Xs=Xs.values.astype(np.float32), y=lab, h2=int(P.EI.load_match(m)['half_start'][1]) // STEP)
            print('  ', m, s, Xd.shape, flush=True)
    np.save(os.path.join(OUT, 'cols.npy'), np.array(cols))
    for m in P.EM.list_matches():
        for s in (0, 1):
            X = P.frames(m, s)
            Xd = X[cols].iloc[::STEP].reset_index(drop=True)
            np.savez_compressed(os.path.join(OUT, 'pred_{}_{}.npz'.format(m, s)), X=Xd.values.astype(np.float32), Xs=P.V17.swap_teams(Xd).values.astype(np.float32), n=len(X), cuts=np.array([0, int(P.EM.load_match(m)['half_start'][1]) // STEP, len(Xd)]))
            print('  ', m, s, Xd.shape, flush=True)
            np.savez_compressed(os.path.join(OUT, 'MET-{}_{}.npz'.format(m, s)), X=Xd.values.astype(np.float32), Xs=P.V17.swap_teams(Xd).values.astype(np.float32), y=X['label'].values[::STEP], h2=int(P.EM.load_match(m)['half_start'][1]) // STEP)
    from src import config as C
    XB = P.barca_X({})
    Xd = XB[cols].iloc[::STEP].reset_index(drop=True)
    (a1, b1), (a2, b2) = (C.HALVES[1], C.HALVES[2])
    np.savez_compressed(os.path.join(OUT, 'pred_barca_0.npz'), X=Xd.values.astype(np.float32), Xs=P.V17.swap_teams(Xd).values.astype(np.float32), n=len(XB), cuts=np.array([0, a1 // STEP, (b1 + 1) // STEP, a2 // STEP, (b2 + 1) // STEP, len(Xd)]))
    print('   barca', Xd.shape, flush=True)
if __name__ == '__main__':
    main()
