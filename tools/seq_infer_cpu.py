import argparse
import os
import sys
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_net(pt):
    ck = torch.load(pt, map_location='cpu', weights_only=False)
    for k, v in ck['cfg'].items():
        os.environ[k] = v
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import importlib
    import seq_model as S
    S = importlib.reload(S)
    net = S.Net(2 * ck['n_in'])
    net.load_state_dict(ck['state_dict'])
    net.eval()
    return (S, net, ck)

def predict_cpu(S, net, X, Xs, cuts, mu, sd):
    pd_, ph = (np.full(len(X), 0.5), np.full(len(X), 0.5))
    with torch.no_grad():
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 10:
                continue
            o1 = torch.sigmoid(net(torch.from_numpy(S.prep(X[a:b], mu, sd))[None]))[0].numpy()
            o2 = torch.sigmoid(net(torch.from_numpy(S.prep(Xs[a:b], mu, sd))[None]))[0].numpy()
            pd_[a:b] = 0.5 * (o1[:, 0] + o2[:, 0])
            ph[a:b] = 0.5 * (o1[:, 1] + 1 - o2[:, 1])
    return (pd_.astype(np.float32), ph.astype(np.float32))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pesos')
    ap.add_argument('salidas_gpu', nargs='?')
    ap.add_argument('--entrada', default=os.path.join(ROOT, 'cache', 'seq', 'pred_barca_0.npz'))
    ap.add_argument('--escribir')
    ap.add_argument('--redondear', type=int)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    S, net, ck = load_net(a.pesos)
    z = np.load(a.entrada)
    ci = S.col_idx()
    X, Xs = (z['X'], z['Xs']) if ci is None else (z['X'][:, ci], z['Xs'][:, ci])
    pd_, ph = predict_cpu(S, net, X, Xs, list(z['cuts']), ck['mu'], ck['sd'])
    if a.redondear is not None:
        pd_, ph = (np.round(pd_, a.redondear), np.round(ph, a.redondear))
    if a.salidas_gpu:
        ref = np.load(a.salidas_gpu)
        for k, v in (('_pd', pd_), ('_ph', ph)):
            r = ref['barca_0' + k] if a.redondear is None else np.round(ref['barca_0' + k], a.redondear)
            d = np.abs(v - r)
            print('Barça{}: diferencia CPU−GPU máx. {:.2e}, media {:.2e}, frames distintos {}'.format(k, d.max(), d.mean(), int((d > 0).sum())))
    if a.escribir:
        out = dict(np.load(a.salidas_gpu)) if a.salidas_gpu else {}
        out['barca_0_pd'], out['barca_0_ph'] = (pd_, ph)
        np.savez_compressed(a.escribir, **out)
        print('->', a.escribir)
if __name__ == '__main__':
    main()
