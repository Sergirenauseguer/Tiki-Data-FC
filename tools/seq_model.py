import glob
import os
import sys
import numpy as np
import torch
import torch.nn as nn
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
SEQ = os.path.join(ROOT, 'cache', 'seq')
OUT = os.path.join(ROOT, 'experiments', 'pipeline_approach_v21', os.environ.get('SEQ_OUT', 'diag_seq.npz'))
STEP = 5
MT = os.environ.get('SEQ_MT') == '1'
STAGES = int(os.environ.get('SEQ_STAGES', 1))
TMSE = float(os.environ.get('SEQ_TMSE', 0.0))
WIN, BATCH, ITERS, HID = (int(os.environ.get('SEQ_WIN', 300)), 32, int(os.environ.get('SEQ_ITERS', 2500)), int(os.environ.get('SEQ_HID', 128)))
torch.manual_seed(0)
np.random.seed(0)

class Refine(nn.Module):

    def __init__(self, k, ch=64, layers=6):
        super().__init__()
        self.inp = nn.Conv1d(k, ch, 1)
        self.dil = nn.ModuleList([nn.Conv1d(ch, ch, 3, padding=2 ** i, dilation=2 ** i) for i in range(layers)])
        self.pw = nn.ModuleList([nn.Conv1d(ch, ch, 1) for _ in range(layers)])
        self.out = nn.Conv1d(ch, k, 1)

    def forward(self, p):
        h = self.inp(p.transpose(1, 2))
        for d, w in zip(self.dil, self.pw):
            h = h + w(torch.relu(d(h)))
        return self.out(h).transpose(1, 2)

class Net(nn.Module):

    def __init__(self, d):
        super().__init__()
        k = 4 if MT else 2
        self.conv = nn.Sequential(nn.Conv1d(d, HID, 5, padding=2), nn.ReLU(), nn.Dropout(0.2), nn.Conv1d(HID, HID, 5, padding=2), nn.ReLU())
        self.gru = nn.GRU(HID, HID, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Linear(2 * HID, k)
        self.stages = nn.ModuleList([Refine(k) for _ in range(STAGES - 1)])

    def forward(self, x, all_stages=False):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        outs = [self.head(h)]
        for st in self.stages:
            outs.append(st(torch.sigmoid(outs[-1])))
        return outs if all_stages else outs[-1]
PLAYER_COLS = ['p90', 'p50', 'pmax', 'p90_c1', 'p90_c3', 'p90_c8', 'p90_p3', 'p90_f3', 'p90_p6', 'p90_f6', 'p90_p1', 'p90_f1', 'p50_c3', 'p90_dfp', 'p90_min_c5', 'n_fast', 'n_fast_c2', 'spd_h', 'spd_a', 'width_h', 'length_h', 'cx_h', 'width_a', 'length_a', 'cx_a', 'n_in_box']

def col_idx():
    if os.environ.get('SEQ_FEATS') != 'players':
        return None
    cols = list(np.load(os.path.join(SEQ, 'cols.npy')))
    return np.array([cols.index(c) for c in PLAYER_COLS])

def load():
    data = {}
    ci = col_idx()
    files = sorted(glob.glob(os.path.join(SEQ, 'DFL-*_[01].npz')))
    if os.environ.get('SEQ_METRICA') == '1':
        files += sorted(glob.glob(os.path.join(SEQ, 'MET-*_[01].npz')))
    for f in files:
        m, s = os.path.basename(f)[:-4].replace('MET-', '').rsplit('_', 1)
        z = np.load(f, allow_pickle=True)
        lab = os.environ.get('SEQ_LABEL')
        if lab == 'c20':
            y = z['y_c20']
        elif lab == 'ref' and 'y_ref' in z.files:
            y = z['y_ref']
        else:
            y = z['y']
        yaux = z['y_aux'].astype(str) if MT and 'y_aux' in z.files else None
        X, Xs = (z['X'], z['Xs']) if ci is None else (z['X'][:, ci], z['Xs'][:, ci])
        data[m, int(s)] = (X, Xs, y.astype(str), int(z['h2']), yaux)
    return data

def prep(X, mu, sd):
    Z = (X - mu) / sd
    nan = np.isnan(Z)
    Z[nan] = 0.0
    return np.concatenate([np.clip(Z, -6, 6), nan.astype(np.float32)], 1).astype(np.float32)

def predict(net, X, Xs, cuts, mu, sd):
    pd_, ph, pc = (np.full(len(X), 0.5), np.full(len(X), 0.5), np.zeros(len(X)))
    with torch.no_grad():
        for a, b in zip(cuts[:-1], cuts[1:]):
            if b - a < 10:
                continue
            o1 = torch.sigmoid(net(torch.from_numpy(prep(X[a:b], mu, sd))[None].to('cuda')))[0].cpu().numpy()
            o2 = torch.sigmoid(net(torch.from_numpy(prep(Xs[a:b], mu, sd))[None].to('cuda')))[0].cpu().numpy()
            pd_[a:b] = 0.5 * (o1[:, 0] + o2[:, 0])
            ph[a:b] = 0.5 * (o1[:, 1] + 1 - o2[:, 1])
            if MT:
                pc[a:b] = 0.5 * (o1[:, 2] + o2[:, 2])
    return (pd_, ph, pc) if MT else (pd_, ph)

def change_soft(y, sigma=1.0):
    idx = np.flatnonzero((y[1:] != y[:-1]) & np.isin(y[1:], ['H', 'A']) & np.isin(y[:-1], ['H', 'A'])) + 1
    t = np.zeros(len(y), np.float32)
    for k in range(-3, 4):
        j = np.clip(idx + k, 0, len(y) - 1)
        t[j] = np.maximum(t[j], np.exp(-k * k / (2 * sigma ** 2)))
    return t

def run_fold(data, test, return_model=False):
    dev = 'cuda'
    train = [k for k in data if k[0] != test]
    allX = np.concatenate([data[k][0] for k in train])
    mu, sd = (np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-06)
    seqs = []
    for k in train:
        X, Xs, y, h2, yaux = data[k]
        dead = (y == 'D').astype(np.float32)
        home = (y == 'H').astype(np.float32)
        chg = change_soft(y) if MT else np.zeros(len(y), np.float32)
        ah = (yaux == 'H').astype(np.float32) if yaux is not None else np.zeros(len(y), np.float32)
        al = np.isin(yaux, ['H', 'A']).astype(np.float32) if yaux is not None else np.zeros(len(y), np.float32)
        for a, b in ((0, h2), (h2, len(y))):
            seqs.append((prep(X[a:b], mu, sd), prep(Xs[a:b], mu, sd), dead[a:b], home[a:b], chg[a:b], ah[a:b], al[a:b]))
    d = seqs[0][0].shape[1]
    net = Net(d).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.0001)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.002, total_steps=ITERS)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    lens = np.array([len(s[2]) for s in seqs], float)
    for it in range(ITERS):
        xb, yd, yh, yc, ya, yl = ([], [], [], [], [], [])
        for _ in range(BATCH):
            i = np.random.choice(len(seqs), p=lens / lens.sum())
            Xa, Xb, dd, hh, cc, aa, ll = seqs[i]
            a = np.random.randint(0, len(dd) - WIN)
            sw = np.random.rand() < 0.5
            xb.append((Xb if sw else Xa)[a:a + WIN])
            yd.append(dd[a:a + WIN])
            yh.append((1 - hh[a:a + WIN]) * (1 - dd[a:a + WIN]) if sw else hh[a:a + WIN])
            yc.append(cc[a:a + WIN])
            ya.append((1 - aa[a:a + WIN]) * ll[a:a + WIN] if sw else aa[a:a + WIN])
            yl.append(ll[a:a + WIN])
        xb = torch.from_numpy(np.stack(xb)).to(dev)
        yd = torch.from_numpy(np.stack(yd)).to(dev)
        yh = torch.from_numpy(np.stack(yh)).to(dev)
        live = 1 - yd
        if MT:
            yc = torch.from_numpy(np.stack(yc)).to(dev)
            ya = torch.from_numpy(np.stack(ya)).to(dev)
            yl = torch.from_numpy(np.stack(yl)).to(dev)
        loss = 0.0
        for out in net(xb, all_stages=True):
            loss = loss + bce(out[..., 0], yd).mean() + (bce(out[..., 1], yh) * live).sum() / live.sum().clamp(min=1)
            if MT:
                loss = loss + (bce(out[..., 2], yc) * live).sum() / live.sum().clamp(min=1) + 0.5 * (bce(out[..., 3], ya) * yl).sum() / yl.sum().clamp(min=1)
            if TMSE > 0:
                lp = nn.functional.logsigmoid(out[..., :2])
                dlt = (lp[:, 1:] - lp[:, :-1].detach()).abs().clamp(max=4.0)
                loss = loss + TMSE * (dlt ** 2).mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 500 == 0:
            print('    it {} loss {:.4f}'.format(it, loss.item()), flush=True)
    net.eval()
    if return_model:
        return (net, mu, sd)
    X, Xs, y, h2 = data[test, 0][:4]
    pd_, ph = (np.zeros(len(y)), np.zeros(len(y)))
    with torch.no_grad():
        for a, b in ((0, h2), (h2, len(y))):
            o1 = torch.sigmoid(net(torch.from_numpy(prep(X[a:b], mu, sd))[None].to(dev)))[0].cpu().numpy()
            o2 = torch.sigmoid(net(torch.from_numpy(prep(Xs[a:b], mu, sd))[None].to(dev)))[0].cpu().numpy()
            pd_[a:b] = 0.5 * (o1[:, 0] + o2[:, 0])
            ph[a:b] = 0.5 * (o1[:, 1] + 1 - o2[:, 1])
    return (pd_, ph)

def main_full():
    data = load()
    ids = sorted({m for m, _ in data})
    path = os.path.join(ROOT, 'experiments', 'pipeline_approach_v21', os.environ.get('SEQ_FULL_OUT', 'seq_full.npz'))
    part = path.replace('.npz', '_parcial.npz')
    out = dict(np.load(part)) if os.path.isfile(part) else {}
    for m in ids + ['__todos__']:
        if m != '__todos__' and m + '_1_ph' in out:
            continue
        torch.manual_seed(int(os.environ.get('SEQ_SEED', 0)))
        np.random.seed(int(os.environ.get('SEQ_SEED', 0)))
        net, mu, sd = run_fold(data, m, return_model=True)
        if m != '__todos__':
            for c in (0, 1):
                X, Xs, y, h2 = data[m, c][:4]
                r = predict(net, X, Xs, [0, h2, len(y)], mu, sd)
                pd_, ph = (r[0], r[1])
                out['{}_{}_pd'.format(m, c)], out['{}_{}_ph'.format(m, c)] = (pd_.astype(np.float32), ph.astype(np.float32))
                if MT:
                    out['{}_{}_pc'.format(m, c)] = r[2].astype(np.float32)
                live = y != 'D'
                print('  {} copia {}: equipo acertado {:.3f}'.format(m, c, np.mean((ph[live] > 0.5) == (y[live] == 'H'))), flush=True)
        else:
            for f in sorted(glob.glob(os.path.join(SEQ, 'pred_*.npz'))):
                z = np.load(f)
                key = os.path.basename(f)[5:-4]
                if key + '_ph' in out:
                    continue
                ci = col_idx()
                zx, zxs = (z['X'], z['Xs']) if ci is None else (z['X'][:, ci], z['Xs'][:, ci])
                r = predict(net, zx, zxs, list(z['cuts']), mu, sd)
                out[key + '_pd'], out[key + '_ph'] = (r[0].astype(np.float32), r[1].astype(np.float32))
                if MT:
                    out[key + '_pc'] = r[2].astype(np.float32)
                print('  predicho', key, flush=True)
        np.savez_compressed(part, **out)
    np.savez_compressed(path, step=STEP, **out)
    print('->', path, flush=True)

def save_final(out_pt, ref_npz=None):
    data = load()
    seed = int(os.environ.get('SEQ_SEED', 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    net, mu, sd = run_fold(data, '__todos__', return_model=True)
    cfg = {k: os.environ.get(k) for k in ('SEQ_SEED', 'SEQ_WIN', 'SEQ_ITERS', 'SEQ_HID', 'SEQ_LABEL', 'SEQ_METRICA', 'SEQ_MT', 'SEQ_FEATS', 'SEQ_STAGES', 'SEQ_TMSE') if os.environ.get(k) is not None}
    torch.save({'state_dict': {k: v.cpu() for k, v in net.state_dict().items()}, 'mu': mu, 'sd': sd, 'cfg': cfg, 'cols': list(np.load(os.path.join(SEQ, 'cols.npy'))), 'n_in': int(len(mu))}, out_pt)
    z = np.load(os.path.join(SEQ, 'pred_barca_0.npz'))
    ci = col_idx()
    zx, zxs = (z['X'], z['Xs']) if ci is None else (z['X'][:, ci], z['Xs'][:, ci])
    r = predict(net, zx, zxs, list(z['cuts']), mu, sd)
    print('pesos ->', out_pt, flush=True)
    if ref_npz:
        ref = np.load(ref_npz)
        for k, i in (('_pd', 0), ('_ph', 1)):
            d = np.abs(r[i] - ref['barca_0' + k])
            print('  Barça{}: diferencia máx. con {} = {:.2e}'.format(k, os.path.basename(ref_npz), d.max()), flush=True)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'full':
        return main_full()
    if len(sys.argv) > 2 and sys.argv[1] == 'save':
        return save_final(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    data = load()
    ids = sorted({m for m, _ in data})
    out = {}
    for m in ids:
        pd_, ph = run_fold(data, m)
        out[m + '_pd'], out[m + '_ph'] = (pd_.astype(np.float32), ph.astype(np.float32))
        y = data[m, 0][2]
        live = y != 'D'
        print('  fuera {}: equipo acertado (vivo, 5 fps) {:.3f}'.format(m, np.mean((ph[live] > 0.5) == (y[live] == 'H'))), flush=True)
    np.savez_compressed(OUT, step=STEP, **out)
    print('->', OUT)
if __name__ == '__main__':
    main()
