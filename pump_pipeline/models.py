"""The three model families. All share one interface:

    model.fit(X, y_fault, y_sev, groups, n_fault, max_sev)   # integer fault classes, severities 0..max_sev
    model.predict(X) -> (fault_pred, sev_pred)
    model.predict_proba(X) -> (P_fault (n, n_fault), P_sev (n, max_sev + 1))   # rows sum to 1
"""
import ctypes
import glob
import logging
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import QuantileTransformer
from sklearn.utils.class_weight import compute_sample_weight

from . import config


def _preload_openmp():
    """xgboost's macOS wheel links @rpath/libomp.dylib (normally Homebrew). Reuse torch's copy so one
    OpenMP runtime serves torch, sklearn and xgboost and no system install is required."""
    try:
        import torch
        libs = glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libomp.dylib"))
        if libs:
            ctypes.CDLL(libs[0], mode=ctypes.RTLD_GLOBAL)
    except ImportError:
        pass


_preload_openmp()
import torch  # noqa: E402
import xgboost as xgb  # noqa: E402

log = logging.getLogger(__name__)


def ordinal_targets(y_sev: np.ndarray, max_sev: int) -> np.ndarray:
    """Severity s -> cumulative binary vector [s>0, s>1, ..., s>max_sev-1]."""
    return (y_sev[:, None] > np.arange(max_sev)[None, :]).astype(np.float32)


def cumulative_to_pmf(C: np.ndarray) -> np.ndarray:
    """Columns P(sev > k), k = 0..S-1  ->  P(sev = s), s = 0..S (monotonicity enforced first)."""
    C = np.minimum.accumulate(np.clip(C, 0.0, 1.0), axis=1)
    ones, zeros = np.ones((len(C), 1)), np.zeros((len(C), 1))
    return np.hstack([ones, C]) - np.hstack([C, zeros])


# =========================================================================== Model 1
class HierarchicalRandomForest:
    """Stage 1: RF over fault types. Stage 2: one RF per fault type over that fault's severity levels,
    selected at inference time by the Stage-1 prediction (so severity is conditioned on the fault)."""

    name = "Hierarchical RF"

    def __init__(self, n_estimators=300, n_jobs=-1, random_state=config.RANDOM_STATE):
        self.kw = dict(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state,
                       class_weight="balanced_subsample", min_samples_leaf=1, max_features="sqrt")

    def fit(self, X, y_fault, y_sev, groups=None, n_fault=None, max_sev=None):
        self.n_fault = n_fault or int(y_fault.max()) + 1
        self.max_sev = max_sev or int(y_sev.max())
        self.stage1 = RandomForestClassifier(**self.kw).fit(X, y_fault)
        self.stage2 = {}
        for f in np.unique(y_fault):
            m = y_fault == f
            levels = np.unique(y_sev[m])
            if len(levels) == 1:                      # single-level fault: severity is implied
                self.stage2[f] = int(levels[0])
            else:
                self.stage2[f] = RandomForestClassifier(**{**self.kw, "n_estimators": 200}).fit(X[m], y_sev[m])
        return self

    def predict(self, X):
        fault = self.stage1.predict(X)
        sev = np.zeros(len(X), dtype=int)
        for f in np.unique(fault):
            m = fault == f
            sub = self.stage2[f]
            sev[m] = sub if isinstance(sub, int) else sub.predict(X[m])
        return fault, sev

    def predict_proba(self, X):
        """P(sev = s) = sum_f P(f) * P(s | f): every Stage-2 model weighted by its fault's probability."""
        Pf = np.zeros((len(X), self.n_fault))
        Pf[:, self.stage1.classes_] = self.stage1.predict_proba(X)
        Ps = np.zeros((len(X), self.max_sev + 1))
        for f, sub in self.stage2.items():
            if isinstance(sub, int):
                Ps[:, sub] += Pf[:, f]
            else:
                Ps[:, sub.classes_] += Pf[:, [f]] * sub.predict_proba(X)
        return Pf, Ps


# =========================================================================== Model 2
class MultiOutputXGBoost:
    """A single native multi-output XGBoost model (multi_strategy='multi_output_tree': every tree has
    vector-valued leaves shared by all targets). Targets = one-hot fault type (K columns) + cumulative
    ordinal encoding of severity (S columns, P(sev > k)). Decoding: fault = argmax over the fault
    block; severity = sum_k P(sev > k), rounded (expected value under the ordinal model)."""

    name = "Multi-Output XGBoost"

    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.1, random_state=config.RANDOM_STATE):
        self.params = dict(
            tree_method="hist", multi_strategy="multi_output_tree", n_estimators=n_estimators,
            max_depth=max_depth, learning_rate=learning_rate, max_bin=64, subsample=0.8,
            colsample_bynode=0.3, min_child_weight=2, reg_lambda=1.0, n_jobs=-1, random_state=random_state,
        )

    def fit(self, X, y_fault, y_sev, groups=None, n_fault=None, max_sev=None):
        self.n_fault = n_fault or int(y_fault.max()) + 1
        self.max_sev = max_sev or int(y_sev.max())
        Y = np.hstack([np.eye(self.n_fault, dtype=np.float32)[y_fault], ordinal_targets(y_sev, self.max_sev)])
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(X, Y, sample_weight=compute_sample_weight("balanced", y_fault))
        return self

    def predict(self, X):
        P = self.model.predict_proba(X)
        fault = P[:, : self.n_fault].argmax(1)
        sev = np.rint(P[:, self.n_fault:].sum(1)).astype(int)
        return fault, sev

    def predict_proba(self, X):
        P = self.model.predict_proba(X)
        Pf = P[:, : self.n_fault]
        return Pf / np.maximum(Pf.sum(1, keepdims=True), 1e-12), cumulative_to_pmf(P[:, self.n_fault:])


# =========================================================================== Model 3
class _MultiTaskNet(torch.nn.Module):
    """Shared MLP trunk -> softmax fault head + CORAL ordinal severity head
    (one shared projection, S ordered bias thresholds => rank-consistent cumulative probabilities)."""

    def __init__(self, d_in, n_fault, max_sev, hidden=(512, 256), dropout=0.3):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.BatchNorm1d(h), torch.nn.GELU(), torch.nn.Dropout(dropout)]
            d = h
        self.trunk = torch.nn.Sequential(*layers)
        self.fault_head = torch.nn.Linear(d, n_fault)
        self.sev_proj = torch.nn.Linear(d, 1, bias=False)
        self.sev_bias = torch.nn.Parameter(torch.linspace(1.0, -1.0, max_sev))

    def forward(self, x):
        h = self.trunk(x)
        return self.fault_head(h), self.sev_proj(h) + self.sev_bias


class MultiTaskMLP:
    name = "Multi-Task MLP"

    def __init__(self, epochs=120, batch_size=256, lr=1e-3, weight_decay=1e-4, patience=15,
                 sev_loss_weight=1.0, random_state=config.RANDOM_STATE):
        self.epochs, self.batch_size, self.lr, self.wd = epochs, batch_size, lr, weight_decay
        self.patience, self.sev_w, self.seed = patience, sev_loss_weight, random_state

    def _tensors(self, X):
        return torch.as_tensor(self.scaler.transform(X), dtype=torch.float32)

    def fit(self, X, y_fault, y_sev, groups, n_fault=None, max_sev=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.n_fault = n_fault or int(y_fault.max()) + 1
        self.max_sev = max_sev or int(y_sev.max())
        # Early-stopping split is grouped by recording, like the outer CV, to avoid optimistic stopping
        tr, va = next(GroupShuffleSplit(1, test_size=0.15, random_state=self.seed).split(X, y_fault, groups))
        self.scaler = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, len(tr)),
                                          random_state=self.seed).fit(X[tr])
        Xt, Xv = self._tensors(X[tr]), self._tensors(X[va])
        ft, fv = torch.as_tensor(y_fault[tr]), torch.as_tensor(y_fault[va])
        st = torch.as_tensor(ordinal_targets(y_sev[tr], self.max_sev))
        sv = torch.as_tensor(ordinal_targets(y_sev[va], self.max_sev))

        counts = np.bincount(y_fault[tr], minlength=self.n_fault).astype(np.float32)
        class_w = torch.as_tensor(np.where(counts > 0, counts.sum() / (self.n_fault * np.maximum(counts, 1)), 0.0),
                                  dtype=torch.float32)
        ce = torch.nn.CrossEntropyLoss(weight=class_w)
        bce = torch.nn.BCEWithLogitsLoss()

        self.net = _MultiTaskNet(X.shape[1], self.n_fault, self.max_sev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        best, best_state, bad = np.inf, None, 0
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(perm), self.batch_size):
                b = perm[i: i + self.batch_size]
                if len(b) < 2:
                    continue
                logit_f, logit_s = self.net(Xt[b])
                loss = ce(logit_f, ft[b]) + self.sev_w * bce(logit_s, st[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            self.net.eval()
            with torch.no_grad():
                lf, ls = self.net(Xv)
                val = (ce(lf, fv) + self.sev_w * bce(ls, sv)).item()
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        self.epochs_trained = epoch + 1
        log.debug("MLP stopped after %d epochs (best val loss %.4f)", self.epochs_trained, best)
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            lf, ls = self.net(self._tensors(X))
        return lf.argmax(1).numpy(), (torch.sigmoid(ls) > 0.5).sum(1).numpy()

    def predict_proba(self, X):
        self.net.eval()
        with torch.no_grad():
            lf, ls = self.net(self._tensors(X))
        return torch.softmax(lf, 1).numpy(), cumulative_to_pmf(torch.sigmoid(ls).numpy())


MODEL_FACTORIES = {
    "hrf": HierarchicalRandomForest,
    "xgb": MultiOutputXGBoost,
    "mlp": MultiTaskMLP,
}
