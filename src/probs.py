"""Out-of-sample predicted probabilities P_hat via K-fold cross-validation.

Two backends:
  * "logreg": multinomial logistic regression on fixed features (fast; default).
  * "cnn"   : a small ResNet trained from scratch on raw images (needs torch and ideally a GPU).
Any model works for CL: it only needs out-of-sample probabilities.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold


def _folds(y: np.ndarray, k: int, seed: int):
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


def oof_probs_logreg(X: np.ndarray, y: np.ndarray, m: int, k: int = 4, seed: int = 0,
                     C: float = 1.0, max_iter: int = 200) -> np.ndarray:
    """Out-of-fold probabilities from standardized-feature logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    P = np.zeros((len(y), m))
    for tr, va in _folds(y, k, seed):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=max_iter)
        clf.fit(scaler.transform(X[tr]), y[tr])
        proba = clf.predict_proba(scaler.transform(X[va]))
        P[np.ix_(va, clf.classes_)] = proba  # robust if a class is absent from a training fold
    return P


def oof_probs_cnn(images: np.ndarray, y: np.ndarray, m: int, k: int = 4, seed: int = 0,
                  epochs: int = 20, batch_size: int = 256, lr: float = 0.1,
                  device: str | None = None) -> np.ndarray:
    """Out-of-fold probabilities from a small ResNet trained from scratch.

    images: uint8 array (n, H, W, 3). Uses random-crop + flip augmentation, SGD + one-cycle LR.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)

    mean = images.reshape(-1, 3).mean(0) / 255.0
    std = images.reshape(-1, 3).std(0) / 255.0
    x_all = torch.from_numpy(((images / 255.0 - mean) / std).astype(np.float32)).permute(0, 3, 1, 2).contiguous()
    y_all = torch.from_numpy(y.astype(np.int64))

    def conv_bn(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))

    class Res(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.a, self.b = conv_bn(c, c), conv_bn(c, c)

        def forward(self, x):
            return x + self.b(self.a(x))

    def make_net():
        return nn.Sequential(
            conv_bn(3, 64), conv_bn(64, 128), nn.MaxPool2d(2), Res(128),
            conv_bn(128, 256), nn.MaxPool2d(2), conv_bn(256, 512), nn.MaxPool2d(2), Res(512),
            nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(512, m),
        )

    def augment(xb):
        n = xb.shape[0]
        flip = torch.rand(n, device=xb.device) < 0.5
        xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        xb = F.pad(xb, (4, 4, 4, 4), mode="reflect")
        i, j = np.random.randint(0, 9, 2)
        return xb[:, :, i:i + images.shape[1], j:j + images.shape[2]]

    P = np.zeros((len(y), m))
    for tr, va in _folds(y, k, seed):
        net = make_net().to(dev)
        opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        steps = epochs * int(np.ceil(len(tr) / batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
        xtr, ytr = x_all[tr], y_all[tr]
        for _ in range(epochs):
            net.train()
            perm = torch.randperm(len(tr))
            for s in range(0, len(tr), batch_size):
                idx = perm[s:s + batch_size]
                xb, yb = augment(xtr[idx].to(dev)), ytr[idx].to(dev)
                loss = F.cross_entropy(net(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        net.eval()
        with torch.no_grad():
            out = [F.softmax(net(x_all[va[s:s + 1024]].to(dev)), dim=1).cpu().numpy()
                   for s in range(0, len(va), 1024)]
        P[va] = np.concatenate(out)
    return P
