"""The downstream classifier of Garcia-Jara et al. (2022), Sec. 4.3 and Table 3.

Base classifier: 5 conv blocks (Conv k=3 + BatchNorm, MaxPool 2 + ReLU), channels 2->32->64->128->64->64,
then Dense -> 100, BN + ReLU + Dropout, Dense -> N. Input = magnitudes and phases (2 channels).
The classifier is an ensemble of 5 base CNNs, and every score is repeated over 15 independent runs,
i.e. 75 small CNNs per table row. We train them side by side as ONE network with a leading "member"
axis: every member has its own weights and its own mini-batches, so they are exactly as independent
as 75 separate trainings. Each layer is one batched matrix product over the members (a convolution
with kernel 3 is a matrix product on the 3 shifted copies of its input), which a GPU runs far faster
than 75 small networks or a grouped convolution (measured on a T4: 133 s per epoch with grouped
convolutions). Mixed precision (fp16) is used on the GPU for the same reason.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sampling import ClassResampler


class MemberLinear(nn.Module):
    """M independent Linear layers. x: (M, B, fin) -> (M, B, fout)."""

    def __init__(self, M, fin, fout):
        super().__init__()
        bound = 1 / math.sqrt(fin)                     # same init as nn.Linear
        self.w = nn.Parameter(torch.empty(M, fin, fout).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.empty(M, 1, fout).uniform_(-bound, bound))

    def forward(self, x):
        return torch.baddbmm(self.b, x, self.w)


class MemberConv(nn.Module):
    """M independent Conv1d(kernel 3, padding 1) layers. x: (M, cin, B, L) -> (M, cout, B, L)."""

    def __init__(self, M, cin, cout, k=3):
        super().__init__()
        self.k = k
        bound = 1 / math.sqrt(cin * k)                 # same init as nn.Conv1d
        self.w = nn.Parameter(torch.empty(M, cout, cin * k).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.empty(M, cout, 1).uniform_(-bound, bound))

    def forward(self, x):
        M, C, B, L = x.shape
        xp = F.pad(x, (self.k // 2, self.k // 2))
        cols = torch.stack([xp[..., i:i + L] for i in range(self.k)], 2)       # (M, cin, k, B, L)
        return torch.baddbmm(self.b, self.w, cols.reshape(M, C * self.k, B * L)).view(M, -1, B, L)


class CNNEnsemble(nn.Module):
    """M copies of the paper's base classifier (Table 3), run side by side. Input (M, B, in_ch, L)."""

    def __init__(self, M, n_classes, L, in_ch=2, dropout=0.5):
        super().__init__()
        self.M, self.in_ch = M, in_ch
        chans = [in_ch, 32, 64, 128, 64, 64]
        self.convs = nn.ModuleList(MemberConv(M, a, b) for a, b in zip(chans[:-1], chans[1:]))
        self.bns = nn.ModuleList(nn.BatchNorm2d(M * b) for b in chans[1:])
        l = L
        for _ in range(5):
            l = math.ceil(l / 2)                       # ceil(L/32) as in Table 3
        self.fc1 = MemberLinear(M, 64 * l, 100)
        self.bn = nn.BatchNorm1d(M * 100)
        self.drop = nn.Dropout(dropout)
        self.fc2 = MemberLinear(M, 100, n_classes)

    def features(self, x):
        """Output of the last conv block, flattened: (M, B, 64 * ceil(L/32))."""
        h = x.transpose(1, 2)                                              # (M, C, B, L)
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h)
            M, C, B, L = h.shape
            h = bn(h.reshape(1, M * C, B, L))                              # statistics over (B, L) per member-channel
            h = F.relu(F.max_pool2d(h, (1, 2), ceil_mode=True)).view(M, C, B, -1)
        return h.transpose(1, 2).reshape(h.shape[0], h.shape[2], -1)

    def forward(self, x):
        h = self.fc1(self.features(x))                                     # (M, B, 100)
        M, B, _ = h.shape
        h = self.bn(h.transpose(1, 2).reshape(1, M * 100, B)).view(M, 100, B).transpose(1, 2)
        return self.fc2(self.drop(F.relu(h)))                              # (M, B, n_classes) logits


def reference_cnn(ens, m):
    """Member m of an ensemble rebuilt from standard PyTorch layers (used only by the self-tests)."""
    layers = []
    for conv in ens.convs:
        cout, cin = conv.w.shape[1], conv.w.shape[2] // conv.k
        ref = nn.Conv1d(cin, cout, conv.k, padding=conv.k // 2)
        ref.weight.data, ref.bias.data = conv.w[m].reshape(cout, cin, conv.k).clone(), conv.b[m, :, 0].clone()
        layers += [ref, nn.BatchNorm1d(cout), nn.MaxPool1d(2, ceil_mode=True), nn.ReLU()]
    fcs = []
    for fc in (ens.fc1, ens.fc2):
        ref = nn.Linear(fc.w.shape[1], fc.w.shape[2])
        ref.weight.data, ref.bias.data = fc.w[m].T.clone(), fc.b[m, 0].clone()
        fcs.append(ref)
    return nn.Sequential(*layers, nn.Flatten(), fcs[0], nn.BatchNorm1d(100), nn.ReLU(), fcs[1])


def stack_input(x, ph):
    """(n, L) magnitudes + (n, L) phases -> (n, 2, L)."""
    return torch.stack([x, ph], dim=1)


def amp_on(amp, device):
    return bool(amp) and str(device).startswith("cuda")


def autocast(device, amp):
    """fp16 mixed precision on the GPU when amp is on; a no-op otherwise (e.g. in the offline CPU tests)."""
    on = amp_on(amp, device)
    return torch.autocast("cuda" if on else "cpu", enabled=on)


@torch.no_grad()
def predict_proba(model, X, batch=512, amp=True):
    """Softmax outputs of every member: (M, n, N). X: (n, 2, L) on the model's device."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].unsqueeze(0).expand(model.M, -1, -1, -1)
        with autocast(X.device.type, amp):
            logits = model(xb)
        out.append(F.softmax(logits.float(), -1))
    return torch.cat(out, 1)


def macro_f1(y, pred, n_classes):
    f1 = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        p = tp / max(np.sum(pred == c), 1)
        r = tp / max(np.sum(y == c), 1)
        f1.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f1))


def fit_ensembles(train, val, test, n_classes, runs=15, ens=5, max_epochs=10, patience=3, batch=128,
                  lr=1e-4, seed=0, warp=None, device="cuda", log=print, tag="", amp=True, checks_per_epoch=4):
    """Train `runs` independent ensembles of `ens` CNNs and score each run on val and test.

    train/val/test: dicts of torch tensors x (n, L), ph (n, L), y (n,). Training batches are class-balanced
    (the resampling block with gamma = 1, one independent stream per member), one epoch = len(train) samples.
    Each run is scored on val and test 4 times per epoch; we keep the check with the lowest ensemble validation
    loss (early stopping) and report its test scores. Measured on Catalina: the validation loss is lowest
    after ~2 epochs and test accuracy then falls from 74% to 67% by epoch 10, because the balanced batches
    repeat the rarest class ~66 times per epoch. So training stops once no run improved for `patience` epochs.
    `warp` optionally augments every training batch (the soft-warping baselines).
    """
    torch.manual_seed(seed)
    M = runs * ens
    L = train["x"].shape[1]
    model = CNNEnsemble(M, n_classes, L).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(enabled=amp_on(amp, device))
    Xtr = stack_input(train["x"], train["ph"]).to(device)
    ytr = train["y"].to(device)
    Xva, yva = stack_input(val["x"], val["ph"]).to(device), val["y"].cpu().numpy()
    Xte, yte = stack_input(test["x"], test["ph"]).to(device), test["y"].cpu().numpy()
    y_np = train["y"].cpu().numpy()
    samplers = [ClassResampler(y_np, gamma=1.0, seed=seed * 1000 + m, n_classes=n_classes) for m in range(M)]
    steps = math.ceil(len(y_np) / batch / checks_per_epoch)            # training steps between two checks

    best_loss = np.full(runs, np.inf)
    best = {"epoch": np.zeros(runs), "val_acc": np.zeros(runs), "test_acc": np.zeros(runs),
            "test_f1": np.zeros(runs), "test_pred": np.zeros((runs, len(yte)), int)}
    curves = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "test_acc": []}
    since = np.zeros(runs, int)
    for check in range(1, int(max_epochs * checks_per_epoch) + 1):
        epoch = check / checks_per_epoch
        model.train()
        idx = torch.from_numpy(np.stack([s.sample(steps * batch) for s in samplers])).to(device)   # (M, S*B)
        tot = torch.zeros((), device=device)                            # summed on the GPU (no sync per step)
        for s in range(steps):
            ib = idx[:, s * batch:(s + 1) * batch]                    # (M, B)
            xb, yb = Xtr[ib], ytr[ib]                                  # (M, B, 2, L), (M, B)
            if warp is not None:
                xb = warp(xb)
            with autocast(device, amp):
                logits = model(xb)
            loss = F.cross_entropy(logits.float().reshape(-1, n_classes), yb.reshape(-1)) * M   # sum over members
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.detach() / M
        tot = tot.item()
        # scores of every run (= average of its members' softmax) on val and test: (runs, n, N)
        pv = predict_proba(model, Xva, amp=amp).reshape(runs, ens, len(yva), n_classes).mean(1).cpu().numpy()
        pt = predict_proba(model, Xte, amp=amp).reshape(runs, ens, len(yte), n_classes).mean(1).cpu().numpy()
        vloss = -np.log(np.take_along_axis(pv, yva[None, :, None], 2)[:, :, 0] + 1e-12).mean(1)   # (runs,)
        vacc = (pv.argmax(2) == yva[None]).mean(1)
        tpred = pt.argmax(2)                                                                    # (runs, n)
        tacc = (tpred == yte[None]).mean(1)
        for r in range(runs):
            if vloss[r] < best_loss[r]:
                best_loss[r], since[r] = vloss[r], 0
                best["epoch"][r], best["val_acc"][r], best["test_acc"][r] = epoch, vacc[r], tacc[r]
                best["test_f1"][r] = macro_f1(yte, tpred[r], n_classes)
                best["test_pred"][r] = tpred[r]
            else:
                since[r] += 1
        curves["epoch"].append(epoch)
        curves["train_loss"].append(tot / steps)
        curves["val_loss"].append(float(vloss.mean()))
        curves["val_acc"].append(float(vacc.mean()))
        curves["test_acc"].append(float(tacc.mean()))
        if check % checks_per_epoch == 0:
            log(f"  {tag} epoch {epoch:.0f}: train loss {tot / steps:.3f} | val loss {vloss.mean():.3f} "
                f"acc {vacc.mean():.3f} | test acc {tacc.mean():.3f}")
        if (since >= patience * checks_per_epoch).all():
            break
    recall = np.array([[np.mean(best["test_pred"][r][yte == c] == c) for c in range(n_classes)]
                       for r in range(runs)])
    return {"test_acc": best["test_acc"].tolist(), "test_f1": best["test_f1"].tolist(),
            "val_acc": best["val_acc"].tolist(), "best_epoch": best["epoch"].tolist(),
            "test_recall_per_class": recall.tolist(), "epochs_run": epoch, "curves": curves}


def train_feature_cnn(train, val, n_classes, max_epochs=20, patience=4, batch=128, lr=1e-4, seed=0,
                      device="cuda", log=print):
    """One base classifier trained on the balanced real training set (D_train^u). Its last conv block is the
    feature space for density, recall, FID and the G-score (Sec. 4.3). Keeps the lowest-val-loss weights.
    (Full precision: it is a single small network and its features are reused by every metric.)"""
    torch.manual_seed(seed)
    L = train["x"].shape[1]
    model = CNNEnsemble(1, n_classes, L).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    Xtr, ytr = stack_input(train["x"], train["ph"]).to(device), train["y"].to(device)
    Xva, yva = stack_input(val["x"], val["ph"]).to(device), val["y"].to(device)
    sampler = ClassResampler(train["y"].cpu().numpy(), gamma=1.0, seed=seed, n_classes=n_classes)
    steps = math.ceil(len(ytr) / batch)
    best, best_state, since, curve = np.inf, None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        idx = torch.from_numpy(sampler.sample(steps * batch)).to(device)
        tot = torch.zeros((), device=device)
        for s in range(steps):
            ib = idx[s * batch:(s + 1) * batch]
            loss = F.cross_entropy(model(Xtr[ib].unsqueeze(0))[0], ytr[ib])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach()
        p = predict_proba(model, Xva, amp=False)[0]
        vloss = F.nll_loss(torch.log(p + 1e-12), yva).item()
        vacc = (p.argmax(1) == yva).float().mean().item()
        curve.append((tot.item() / steps, vloss, vacc))
        if vloss < best:
            best, since = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if since >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    log(f"feature CNN: {epoch} epochs, best val loss {best:.3f}, val acc at best "
        f"{[c[2] for c in curve][int(np.argmin([c[1] for c in curve]))]:.3f}")
    return model, curve


@torch.no_grad()
def extract_features(model, x, ph, batch=1024):
    """Feature vectors (n, 64 * ceil(L/32)) of light curves x, ph (torch tensors on the model's device)."""
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        out.append(model.features(stack_input(x[i:i + batch], ph[i:i + batch]).unsqueeze(0))[0])
    return torch.cat(out)
