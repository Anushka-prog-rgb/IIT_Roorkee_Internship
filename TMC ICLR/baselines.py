"""
Baseline evidential multi-view classifiers, ported from
https://github.com/OverfitFlow/Trust4Conflict (ICML 2025 "Navigating
Conflicting Views: Harnessing Trust for Learning", and the CCML/ECML
baselines it bundles for its own comparisons) so they run against the same
`datasets/*.mat` files and the same `run_pipeline`-style harness as our TMC
model in `train.py`.

The upstream repo's image+text models (TrustEnd2End_Food/*) hardcode a
resnet50 image encoder and a BERT text encoder for exactly 2 modalities --
not applicable here, since our datasets are already-extracted tabular
feature vectors with anywhere from 2 to 12 views. What's ported instead is
each method's fusion/loss *mechanism*, generalized from 2 views to n views
via a plain per-view `Classifier` (Linear + Softplus, identical to the one
in Trust/models/util.py), the same swap our own TMC datasets already made
unnecessary by not needing a raw-pixel encoder in the first place.

Included:
  - ETMC : per-view classifiers + a joint "all-views-concatenated" classifier,
           Dempster-Shafer fused. (no trust discount)
  - TF   : per-view classifiers + a learned Trust Discount reference net that
           demotes each view's evidence by an estimated P(view correct | x_v),
           DS fused. This is the ICML 2025 paper's non-evolutionary variant.
  - ETF  : TF + the joint classifier from ETMC as an extra discounted "view".
           This is the ICML 2025 paper's headline method.
  - CCML : splits each view's evidence into a "common" (min across views) and
           "distinctive" (per-view residual) component, sharpens the common
           component, and recombines -- a different fusion algebra from DS.
  - ECML : per-view classifiers + average fusion + a pairwise conflict/
           divergence regularizer (get_dc_loss) added to the training loss.

Deliberate deviation from the upstream training protocol: the original
`Trust/train.py` repeats each run 10x with fresh splits and, within each of
its 4 stages, keeps whichever epoch's checkpoint scores highest on the *test*
set. That's test-set-informed model selection -- fine for their own
leaderboard, but it would give these baselines an unfair advantage in a
same-conditions comparison against `train.run_pipeline`, which never looks at
the test set until the final, single evaluation. So here every method trains
for a fixed epoch budget and is evaluated once at the end (the reported
model/metrics are always the final-epoch ones), matching our own harness's
protocol exactly. A validation slice carved out of the *training* split
(never the test set) additionally scores every epoch purely so the
best-by-validation checkpoint can be saved to disk alongside the final-epoch
one -- see `_fit_track_best` and `train.BestTracker`/`train.split_val_indices`.
"""
import contextlib
import copy
import io
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from train import load_datasets, DEVICE, BestTracker, split_val_indices, save_checkpoint

CLIP_GRAD_NORM = 5.0


# ============================== shared building blocks ==============================

class Classifier(nn.Module):
    """Per-view evidence head: raw features -> non-negative evidence per class."""
    def __init__(self, in_dim, classes):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, classes), nn.Softplus())

    def forward(self, x):
        return self.fc(x)


class Feater(nn.Module):
    """Bilinear head used by the Trust Discount reference net: combines a
    view's raw features with its own belief mass to score P(view correct)."""
    def __init__(self, in_dims, out_dim):
        super().__init__()
        self.fc = nn.Bilinear(in_dims[0], in_dims[1], out_dim)

    def forward(self, x1, x2):
        return F.softplus(self.fc(x1, x2))


class Fusion:
    """Dempster-Shafer-style evidence fusion, folded left-to-right over any
    number of views (unmodified from Trust4Conflict -- pure tensor math, no
    view-count assumption)."""
    def __init__(self, n_classes, fuse_method="bcf"):
        self.n_classes = n_classes
        self.fuse_two = {
            "bcf": self.constraint_fuse,
            "wbf": self.weighted_avg_fuse,
            "a-cbf": self.cumm_fuse,
            "abf": self.avg_fuse,
        }[fuse_method]

    def constraint_fuse(self, evidence1, evidence2):
        return evidence1 + evidence2 + (evidence1 * evidence2) / evidence1.shape[1]

    def weighted_avg_fuse(self, evidence1, evidence2):
        alpha1 = evidence1 + 1
        u1 = self.n_classes / torch.sum(alpha1, dim=1, keepdim=True)
        alpha2 = evidence2 + 1
        u2 = self.n_classes / torch.sum(alpha2, dim=1, keepdim=True)
        return ((1 - u1) * evidence1 + (1 - u2) * evidence2) / (2 - u1 - u2)

    def avg_fuse(self, evidence1, evidence2):
        return (evidence1 + evidence2) / 2

    def cumm_fuse(self, evidence1, evidence2):
        return evidence1 + evidence2

    def fuse(self, evidence):
        for v in range(len(evidence) - 1):
            evidence_a = self.fuse_two(evidence[0], evidence[1]) if v == 0 else self.fuse_two(evidence_a, evidence[v + 1])
        return evidence_a

    def __call__(self, *args, **kwargs):
        return self.fuse(*args, **kwargs)


def KL(alpha, c):
    beta = torch.ones((1, c), device=alpha.device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    return torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return A + B


def sce_loss(p, alpha, c, smooth_factor=0.6):
    """Smoothed CE against a binary (correct/incorrect) target -- used to
    warm-start the Trust Discount reference net."""
    S = torch.sum(alpha, dim=1, keepdim=True)
    label = F.one_hot(p, num_classes=c)
    label_s = label * smooth_factor + (1 - smooth_factor) / c
    return torch.sum(label_s * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)


def get_dc_loss(evidences, device):
    """Pairwise view-disagreement penalty (ECML): high when confident views
    disagree, ~0 when views are uncertain or already agree."""
    num_views = len(evidences)
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S, dim=-1)
    dc_sum = 0
    for i in range(num_views):
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / max(num_views - 1, 1)
        cc = (1 - u[i]) * (1 - u)
        dc_sum = dc_sum + torch.sum(pd * cc, dim=0)
    return torch.mean(dc_sum)


def uncertainty_auroc(alpha_a, n_classes, correct_mask):
    """AUROC of vacuity (1 - confidence) as a correctness detector: can the
    model's own uncertainty tell you when it's about to be wrong?"""
    u = (n_classes / torch.sum(alpha_a, dim=1)).cpu().numpy()
    incorrect = (~correct_mask).astype(int)
    if len(np.unique(incorrect)) == 1:
        return float("nan")
    return roc_auc_score(incorrect, u)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


# ============================== TrustNet / ETrustNet ==============================

class TrustNet(nn.Module):
    """Per-view evidence heads, one Classifier (or Feater, for the bilinear
    reference net) per view."""
    def __init__(self, n_views, in_dims, out_dim, bi=False):
        super().__init__()
        self.n_views = n_views
        self.bi = bi
        self.clfs = nn.ModuleList(
            [Feater(in_dims[i], out_dim) if bi else Classifier(in_dims[i], out_dim) for i in range(n_views)]
        )

    def forward(self, inp):
        evidence = {}
        for v in range(self.n_views):
            evidence[v] = self.clfs[v](inp[v][0], inp[v][1]) if self.bi else self.clfs[v](inp[v])
        return evidence


class ETrustNet(nn.Module):
    """TrustNet plus one extra head for the joint (all-views-concatenated) classifier."""
    def __init__(self, n_views, in_dims, out_dim, bi=False):
        super().__init__()
        self.n_views = n_views
        self.bi = bi
        assert len(in_dims) == n_views + 1
        self.clfs = nn.ModuleList(
            [Feater(in_dims[i], out_dim) if bi else Classifier(in_dims[i], out_dim) for i in range(n_views)]
        )
        self.clf_ps = Feater(in_dims[-1], out_dim) if bi else Classifier(in_dims[-1], out_dim)

    def forward(self, inp):
        evidence = {}
        for v in range(self.n_views):
            evidence[v] = self.clfs[v](inp[v][0], inp[v][1]) if self.bi else self.clfs[v](inp[v])
        evidence[self.n_views] = self.clf_ps(inp[self.n_views][0], inp[self.n_views][1]) if self.bi else self.clf_ps(inp[self.n_views])
        return evidence


# ============================== TF / ETF (+ ETMC as ETF w/o discount) ==============================

class _TrustDiscountBase(nn.Module):
    """Shared discount/refer-feature logic between TF and ETF -- they differ
    only in whether func_net includes the extra joint-concat "view"."""

    def discount(self, func_evidence, refer_evidence):
        disc_evidence = {}
        for v in range(len(func_evidence)):
            func_evidence_v = func_evidence[v]
            func_s_v = torch.sum(func_evidence_v + 1, dim=1, keepdim=True)
            func_u_v = func_evidence_v.shape[1] / func_s_v

            refer_alpha_v = refer_evidence[v] + 1
            refer_s_v = torch.sum(refer_alpha_v, dim=1, keepdim=True)
            refer_p_v = refer_alpha_v[:, 1:2] / refer_s_v

            disc_evidence[v] = refer_p_v * func_u_v / (1 - refer_p_v + refer_p_v * func_u_v) * func_evidence_v
        return disc_evidence

    def form_refer_feat(self, feat, func_evidence):
        refer_feat = {}
        for v in range(len(func_evidence)):
            func_s_v = torch.sum(func_evidence[v] + 1, dim=1, keepdim=True)
            func_b_v = func_evidence[v] / func_s_v.expand(func_evidence[v].shape)
            refer_feat[v] = (feat[v].detach(), func_b_v.detach())
        return refer_feat

    def init_refer_net(self):
        raise NotImplementedError


class TF(_TrustDiscountBase):
    def __init__(self, n_views, n_classes, views_dims):
        super().__init__()
        self.n_views = n_views
        self.n_classes = n_classes
        self.views_dims = views_dims
        self.fuse = Fusion(n_classes, "bcf")
        self.func_net = TrustNet(n_views, views_dims, n_classes)
        self.refer_net = None

    def init_refer_net(self):
        refer_in_dims = [(self.views_dims[i], self.n_classes) for i in range(self.n_views)]
        self.refer_net = TrustNet(self.n_views, refer_in_dims, 2, bi=True)

    def forward(self, X):
        func_feat = {v: X[v] for v in range(self.n_views)}
        func_evidence = self.func_net(func_feat)
        func_alpha = {k: v + 1 for k, v in func_evidence.items()}

        if self.refer_net is not None:
            refer_feat = self.form_refer_feat(func_feat, func_evidence)
            refer_evidence = self.refer_net(refer_feat)
            refer_alpha = {k: v + 1 for k, v in refer_evidence.items()}
            disc_evidence = self.discount(func_evidence, refer_evidence)
            disc_alpha = {k: v + 1 for k, v in disc_evidence.items()}
            evidence_a = self.fuse(disc_evidence)
        else:
            refer_alpha, disc_alpha = None, None
            evidence_a = self.fuse(func_evidence)

        return func_alpha, refer_alpha, disc_alpha, evidence_a + 1


class ETF(_TrustDiscountBase):
    """TF + a joint classifier over the concatenation of all views, treated
    as an (n_views+1)-th evidence source. With refer_net=None this is ETMC."""
    def __init__(self, n_views, n_classes, views_dims):
        super().__init__()
        self.n_views = n_views
        self.n_classes = n_classes
        self.views_dims = views_dims
        self.fuse = Fusion(n_classes, "bcf")
        func_in_dims = views_dims + [sum(views_dims)]
        self.func_net = ETrustNet(n_views, func_in_dims, n_classes)
        self.refer_net = None

    def init_refer_net(self):
        refer_in_dims = [(self.views_dims[i], self.n_classes) for i in range(self.n_views)]
        refer_in_dims.append((sum(self.views_dims), self.n_classes))
        self.refer_net = ETrustNet(self.n_views, refer_in_dims, 2, bi=True)

    def forward(self, X):
        func_feat = {v: X[v] for v in range(self.n_views)}
        func_feat[self.n_views] = torch.cat([X[v] for v in range(self.n_views)], dim=1)
        func_evidence = self.func_net(func_feat)
        func_alpha = {k: v + 1 for k, v in func_evidence.items()}

        if self.refer_net is not None:
            refer_feat = self.form_refer_feat(func_feat, func_evidence)
            refer_evidence = self.refer_net(refer_feat)
            refer_alpha = {k: v + 1 for k, v in refer_evidence.items()}
            disc_evidence = self.discount(func_evidence, refer_evidence)
            disc_alpha = {k: v + 1 for k, v in disc_evidence.items()}
            evidence_a = self.fuse(disc_evidence)
        else:
            refer_alpha, disc_alpha = None, None
            evidence_a = self.fuse(func_evidence)

        return func_alpha, refer_alpha, disc_alpha, evidence_a + 1


# ============================== CCML ==============================

class CCML(nn.Module):
    """Splits each view's evidence into a common component (elementwise min
    across views) and a distinctive component (per-view residual), sharpens
    the common component (power beta), and recombines -- an alternative to
    Dempster-Shafer fusion aimed at conflicting (not just noisy) views."""
    def __init__(self, n_views, n_classes, views_dims, beta=1.25):
        super().__init__()
        self.n_views = n_views
        self.n_classes = n_classes
        self.beta = beta
        self.clfs = nn.ModuleList([Classifier(views_dims[v], n_classes) for v in range(n_views)])

    def forward(self, X):
        evidences = {v: self.clfs[v](X[v]) for v in range(self.n_views)}
        alpha = {v: evidences[v] + 1 for v in range(self.n_views)}
        alpha_a, alpha_con, alpha_div = self.evidence_dc(alpha)
        return evidences, alpha_a - 1, alpha_con - 1, alpha_div - 1

    def evidence_dc(self, alpha):
        E = {v: torch.nan_to_num(alpha[v] - 1, 0) for v in range(len(alpha))}

        E_con = E[0]
        for v in range(1, len(alpha)):
            E_con = torch.min(E_con, E[v])
        # residualize each view against the common component *before*
        # averaging -- E_div must be the mean *distinctive* evidence
        # (E[v] - E_con), not the mean raw evidence, otherwise the common
        # part gets counted once via E_con and a second time baked into
        # E_div, inflating the fused evidence.
        E = {v: E[v] - E_con for v in range(len(alpha))}
        alpha_con = E_con + 1

        E_div = E[0]
        for v in range(1, len(alpha)):
            E_div = E_div + E[v]
        E_div = E_div / len(alpha)

        S_con = torch.sum(alpha_con, dim=1, keepdim=True)
        b_con = E_con / S_con
        S_b = torch.sum(b_con, dim=1, keepdim=True)
        b_con2 = torch.pow(b_con, self.beta)
        S_b2 = torch.sum(b_con2, dim=1, keepdim=True)
        b_cona = b_con2 * (S_b / S_b2)
        E_con = b_cona * S_con * len(alpha)

        E_a = E_con + E_div
        alpha_a = torch.nan_to_num(E_a + 1, 0)
        alpha_con = torch.nan_to_num(E_con + 1, 0)
        alpha_div = torch.nan_to_num(E_div + 1, 0)
        return alpha_a, alpha_con, alpha_div


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True) - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (alpha - ones).mul(torch.digamma(alpha) - torch.digamma(sum_alpha)).sum(dim=1, keepdim=True)
    return first_term + second_term


def kl_loss(alpha, y, epoch_num, num_classes, annealing_step, device):
    annealing_coef = min(1.0, epoch_num / annealing_step)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    return torch.mean(annealing_coef * kl_divergence(kl_alpha, num_classes, device))


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    S = torch.sum(alpha, dim=1, keepdim=True)
    A = torch.sum(target * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    return torch.mean(A)


def ccml_loss(evidences, evidence_a, evidence_con, evidence_div, target, epoch_num, num_classes,
              annealing_step, gamma, delta, device):
    target_oh = F.one_hot(target, num_classes)
    alpha_con = evidence_con + 1
    alpha_div = evidence_div + 1
    loss = 0
    for v in range(len(evidences)):
        alpha = evidences[v] + 1
        loss = loss + edl_digamma_loss(alpha, target_oh, epoch_num, num_classes, annealing_step, device)
        loss = loss + kl_loss(alpha, target_oh, epoch_num, num_classes, annealing_step, device)
    loss = loss + edl_digamma_loss(alpha_con, target_oh, epoch_num, num_classes, annealing_step, device)
    loss = loss + gamma * edl_digamma_loss(alpha_div, target_oh, epoch_num, num_classes, annealing_step, device)
    loss = loss + delta * kl_loss(alpha_con, target_oh, epoch_num, num_classes, annealing_step, device)
    return loss


# ============================== ECML ==============================

class ECML(nn.Module):
    """Per-view classifiers + average DS-style fusion + a pairwise
    disagreement regularizer (get_dc_loss) added at training time."""
    def __init__(self, n_views, n_classes, views_dims):
        super().__init__()
        self.n_views = n_views
        self.n_classes = n_classes
        self.clfs = nn.ModuleList([Classifier(views_dims[v], n_classes) for v in range(n_views)])
        self.fuse = Fusion(n_classes, "abf")

    def forward(self, X):
        evidences = {v: self.clfs[v](X[v]) for v in range(self.n_views)}
        evidence_a = self.fuse(evidences)
        return evidences, evidence_a


# ============================== training / eval ==============================

def _to_device(X, y, device):
    for v in range(len(X)):
        X[v] = Variable(X[v].to(device))
    return X, Variable(y.long().to(device))


def train_warmup(model, loader, refer_opt, epoch, smooth_factor, n_classes, device):
    model.train()
    for X, y in loader:
        X, y = _to_device(X, y, device)
        func_alpha, refer_alpha, *_ = model(X)
        refer_opt.zero_grad()
        loss = 0
        for v in range(len(func_alpha)):
            t_v = (torch.argmax(func_alpha[v], 1) == y).long()
            loss = loss + sce_loss(t_v, refer_alpha[v], 2, smooth_factor)
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(model.refer_net.parameters(), CLIP_GRAD_NORM)
        refer_opt.step()


def train_func(model, loader, func_opt, epoch, n_classes, annealing_step, device):
    model.train()
    for X, y in loader:
        X, y = _to_device(X, y, device)

        func_alpha, *_ = model(X)
        func_opt.zero_grad()
        loss = sum(ce_loss(y, func_alpha[v], n_classes, epoch, annealing_step) for v in range(len(func_alpha)))
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(model.func_net.parameters(), CLIP_GRAD_NORM)
        func_opt.step()

        # separately train the joint DS-fused prediction (valid whether or
        # not a discount net is in the loop -- fuse() works on plain
        # func_evidence too)
        *_, alpha_a = model(X)
        func_opt.zero_grad()
        loss = ce_loss(y, alpha_a, n_classes, epoch, annealing_step)
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(model.func_net.parameters(), CLIP_GRAD_NORM)
        func_opt.step()


def train_refer(model, loader, refer_opt, epoch, n_classes, annealing_step, device):
    model.train()
    for X, y in loader:
        X, y = _to_device(X, y, device)
        *_, alpha_a = model(X)
        refer_opt.zero_grad()
        loss = ce_loss(y, alpha_a, n_classes, epoch, annealing_step)
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(model.refer_net.parameters(), CLIP_GRAD_NORM)
        refer_opt.step()


def eval_tf(model, loader, epoch, n_classes, annealing_step, device):
    model.eval()
    loss_meter, correct, total = AverageMeter(), 0, 0
    all_u, all_correct = [], []
    for X, y in loader:
        X, y = _to_device(X, y, device)
        with torch.no_grad():
            func_alpha, *_, alpha_a = model(X)
        loss = ce_loss(y, alpha_a, n_classes, epoch, annealing_step)
        for v in range(len(func_alpha)):
            loss = loss + ce_loss(y, func_alpha[v], n_classes, epoch, annealing_step)
        loss_meter.update(loss.mean().item(), y.shape[0])
        pred = torch.argmax(alpha_a, 1)
        correct_mask = (pred == y).cpu().numpy().astype(bool)
        correct += correct_mask.sum()
        total += y.shape[0]
        all_u.append(alpha_a)
        all_correct.append(correct_mask)
    alpha_cat = torch.cat(all_u, dim=0)
    correct_cat = np.concatenate(all_correct)
    return loss_meter.avg, correct / total, uncertainty_auroc(alpha_cat, n_classes, correct_cat)


def train_ccml_epoch(model, loader, optimizer, epoch, n_classes, annealing_step, gamma, delta, device):
    model.train()
    for X, y in loader:
        X, y = _to_device(X, y, device)
        evidences, evidence_a, evidence_con, evidence_div = model(X)
        loss = ccml_loss(evidences, evidence_a, evidence_con, evidence_div, y, epoch, n_classes,
                          annealing_step, gamma, delta, device).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()


def eval_ccml(model, loader, n_classes, epoch, annealing_step, device):
    model.eval()
    correct, total = 0, 0
    loss_meter = AverageMeter()
    all_alpha, all_correct = [], []
    for X, y in loader:
        X, y = _to_device(X, y, device)
        with torch.no_grad():
            evidences, evidence_a, evidence_con, evidence_div = model(X)
        alpha_a = evidence_a + 1
        pred = torch.argmax(alpha_a, 1)
        correct_mask = (pred == y).cpu().numpy().astype(bool)
        correct += correct_mask.sum()
        total += y.shape[0]
        # same EDL-family loss (ce_loss) other methods report, for an
        # apples-to-apples test_loss column in the comparison table --
        # plain softmax cross-entropy on evidence_a isn't what CCML was
        # trained to minimize (ccml_loss) and isn't comparable to TF/ETF/
        # ECML/TMC's reported losses either.
        loss_meter.update(ce_loss(y, alpha_a, n_classes, epoch, annealing_step).mean().item(), y.shape[0])
        all_alpha.append(alpha_a)
        all_correct.append(correct_mask)
    alpha_cat = torch.cat(all_alpha, dim=0)
    correct_cat = np.concatenate(all_correct)
    return loss_meter.avg, correct / total, uncertainty_auroc(alpha_cat, n_classes, correct_cat)


def train_ecml_epoch(model, loader, optimizer, epoch, n_classes, annealing_step, device):
    model.train()
    for X, y in loader:
        X, y = _to_device(X, y, device)
        evidences, evidence_a = model(X)
        alpha_a = evidence_a + 1
        loss = ce_loss(y, alpha_a, n_classes, epoch, annealing_step).mean()
        for v in range(len(evidences)):
            loss = loss + ce_loss(y, evidences[v] + 1, n_classes, epoch, annealing_step).mean()
        loss = loss / (len(evidences) + 1)
        loss = loss + get_dc_loss(evidences, device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
        optimizer.step()


def eval_ecml(model, loader, n_classes, epoch, annealing_step, device):
    model.eval()
    correct, total = 0, 0
    loss_meter = AverageMeter()
    all_alpha, all_correct = [], []
    for X, y in loader:
        X, y = _to_device(X, y, device)
        with torch.no_grad():
            evidences, evidence_a = model(X)
        alpha_a = evidence_a + 1
        pred = torch.argmax(alpha_a, 1)
        correct_mask = (pred == y).cpu().numpy().astype(bool)
        correct += correct_mask.sum()
        total += y.shape[0]
        loss_meter.update(ce_loss(y, alpha_a, n_classes, epoch, annealing_step).mean().item(), y.shape[0])
        all_alpha.append(alpha_a)
        all_correct.append(correct_mask)
    alpha_cat = torch.cat(all_alpha, dim=0)
    correct_cat = np.concatenate(all_correct)
    return loss_meter.avg, correct / total, uncertainty_auroc(alpha_cat, n_classes, correct_cat)


# ============================== dispatcher ==============================

BASELINE_NAMES = ["ETMC", "TF", "ETF", "CCML", "ECML"]


def _fit_track_best(train_step, eval_fn, model, tracker, max_epochs, start_epoch=1, desc=None, progress=True):
    """
    Runs `train_step(epoch)` for exactly `max_epochs` epochs (a fixed
    budget, not a cap) -- whatever `model` looks like after the last epoch
    is what every baseline's training stage(s) end up using. After each
    epoch, `eval_fn(epoch) -> score` scores the *whole* model on the
    held-out validation split and `tracker` (a shared BestTracker, so a
    checkpoint from an earlier stage can still be "best" overall) records
    a snapshot if it improved -- purely for later comparison, never fed
    back into training or used to stop early.
    """
    epoch_range = range(start_epoch, start_epoch + max_epochs)
    epoch_bar = tqdm(epoch_range, desc=desc or 'training', unit='epoch', leave=False) if progress else epoch_range
    for epoch in epoch_bar:
        train_step(epoch)
        try:
            score = eval_fn(epoch)
        except (ValueError, RuntimeError):
            # CCML in particular can produce transiently non-finite evidence
            # (a pre-existing numerical-stability issue in its power-sharpening
            # step, not fixed here) -- skip scoring this epoch for the best-
            # checkpoint tracker rather than aborting the whole run over it.
            # The final evaluation below is unguarded, so a run that's
            # actually broken still surfaces as a failure to the caller.
            if progress:
                epoch_bar.set_postfix(val_acc='n/a', best='%.3f' % tracker.best_score)
            continue
        tracker.step(score, model, epoch)
        if progress:
            epoch_bar.set_postfix(val_acc='%.3f' % score, best='%.3f' % tracker.best_score)


def run_baseline_pipeline(args, model_name, device=DEVICE, verbose=True, progress=True):
    """
    Trains `model_name` (one of BASELINE_NAMES) on args.data_path and
    returns (model, metrics) -- metrics has the same keys as
    train.run_pipeline's, so both can feed the same comparison table.

    A stratified slice of the training set (args.val_size, default 15%,
    never trained on) scores every epoch across every stage so the
    best-by-validation snapshot of the *whole* model can be saved to disk
    alongside the final-epoch one under args.checkpoint_dir -- this doesn't
    change which model is reported as 'the' result (still the final epoch);
    'best_*' metrics are informational.
    """
    if model_name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline {model_name!r}, expected one of {BASELINE_NAMES}")

    t0 = time.time()
    train_dataset, test_dataset = load_datasets(args)
    val_size = getattr(args, 'val_size', 0.15)
    train_idx, val_idx = split_val_indices(train_dataset, val_size=val_size)
    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(train_dataset, val_idx), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    views_dims = [d[0] for d in args.dims]
    n_views, n_classes = args.views, args.classes
    annealing_step = args.lambda_epochs
    if verbose:
        print(f"[{model_name}/{args.data_name}] views={n_views}, dims={views_dims}, classes={n_classes}, "
              f"train={len(train_dataset)} (core={len(train_idx)}, val={len(val_idx)}), test={len(test_dataset)}")

    sink = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()

    use_discount = model_name in ("TF", "ETF")
    n_stages = 3 if use_discount else 1
    stage_bar = tqdm(total=n_stages, desc=f'[{model_name}/{args.data_name}] pipeline', unit='stage') if progress else None
    tracker = BestTracker()

    def _next_stage(desc):
        if stage_bar is not None:
            stage_bar.set_description(f'[{model_name}/{args.data_name}] {desc}')

    with sink:
        if model_name in ("ETMC", "TF", "ETF"):
            model_cls = ETF if model_name in ("ETMC", "ETF") else TF
            model = model_cls(n_views, n_classes, views_dims).to(device)

            if use_discount:
                model.init_refer_net()
                model.to(device)
                refer_opt = optim.Adam(model.refer_net.parameters(), lr=args.rlr, weight_decay=1e-4)
                for epoch in range(1, args.warm_epochs + 1):
                    train_warmup(model, train_loader, refer_opt, epoch, args.smooth_factor, n_classes, device)

            eval_val = lambda e: eval_tf(model, val_loader, e, n_classes, annealing_step, device)[1]

            _next_stage('func_net')
            func_opt = optim.Adam(model.func_net.parameters(), lr=args.lr, weight_decay=1e-4)
            _fit_track_best(lambda e: train_func(model, train_loader, func_opt, e, n_classes, annealing_step, device),
                             eval_val, model, tracker, args.epochs,
                             desc=f"[{model_name}/{args.data_name}] func_net", progress=progress)
            if stage_bar is not None:
                stage_bar.update(1)

            if use_discount:
                _next_stage('refer_net')
                refer_opt = optim.Adam(model.refer_net.parameters(), lr=args.rlr, weight_decay=1e-4)
                _fit_track_best(lambda e: train_refer(model, train_loader, refer_opt, e, n_classes, annealing_step, device),
                                 eval_val, model, tracker, args.epochs,
                                 desc=f"[{model_name}/{args.data_name}] refer_net", progress=progress)
                if stage_bar is not None:
                    stage_bar.update(1)

                _next_stage('func_net (refined)')
                func_opt = optim.Adam(model.func_net.parameters(), lr=args.lr, weight_decay=1e-4)
                _fit_track_best(lambda e: train_func(model, train_loader, func_opt, e, n_classes, annealing_step, device),
                                 eval_val, model, tracker, args.epochs,
                                 desc=f"[{model_name}/{args.data_name}] func_net (refined)", progress=progress)
                if stage_bar is not None:
                    stage_bar.update(1)

            test_loss, test_acc, test_uauroc = eval_tf(model, test_loader, args.epochs, n_classes, annealing_step, device)
            best_eval = lambda: eval_tf(model, test_loader, tracker.best_epoch, n_classes, annealing_step, device)

        elif model_name == "CCML":
            model = CCML(n_views, n_classes, views_dims).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            _fit_track_best(
                lambda e: train_ccml_epoch(model, train_loader, optimizer, e, n_classes, annealing_step,
                                            gamma=1, delta=1, device=device),
                lambda e: eval_ccml(model, val_loader, n_classes, e, annealing_step, device)[1],
                model, tracker, args.epochs, start_epoch=0,
                desc=f"[{model_name}/{args.data_name}]", progress=progress)
            if stage_bar is not None:
                stage_bar.update(1)
            test_loss, test_acc, test_uauroc = eval_ccml(model, test_loader, n_classes, args.epochs, annealing_step, device)
            best_eval = lambda: eval_ccml(model, test_loader, n_classes, tracker.best_epoch, annealing_step, device)

        elif model_name == "ECML":
            model = ECML(n_views, n_classes, views_dims).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            _fit_track_best(
                lambda e: train_ecml_epoch(model, train_loader, optimizer, e, n_classes, annealing_step, device),
                lambda e: eval_ecml(model, val_loader, n_classes, e, annealing_step, device)[1],
                model, tracker, args.epochs,
                desc=f"[{model_name}/{args.data_name}]", progress=progress)
            if stage_bar is not None:
                stage_bar.update(1)
            test_loss, test_acc, test_uauroc = eval_ecml(model, test_loader, n_classes, args.epochs, annealing_step, device)
            best_eval = lambda: eval_ecml(model, test_loader, n_classes, tracker.best_epoch, annealing_step, device)

        # last-epoch model: whatever `model` holds right now (unchanged).
        last_state = copy.deepcopy(model.state_dict())

        # briefly swap in the best-by-validation snapshot to get its test
        # accuracy too, then restore last_state so the returned `model` --
        # and test_loss/test_acc above -- are still the final-epoch model.
        best_test_loss, best_test_acc = None, None
        if tracker.best_state is not None:
            model.load_state_dict(tracker.best_state)
            best_test_loss, best_test_acc = best_eval()[:2]
            model.load_state_dict(last_state)

    if stage_bar is not None:
        stage_bar.close()

    checkpoint_dir = getattr(args, 'checkpoint_dir', 'checkpoints')
    last_path = save_checkpoint(last_state, checkpoint_dir, f'{args.data_name}_{model_name}_last.pt')
    best_path = (save_checkpoint(tracker.best_state, checkpoint_dir, f'{args.data_name}_{model_name}_best.pt')
                 if tracker.best_state is not None else None)

    elapsed = time.time() - t0
    metrics = {
        "dataset": args.data_name,
        "model": model_name,
        "views": n_views,
        "classes": n_classes,
        "dims": views_dims,
        "n_train": len(train_dataset),
        "n_val": len(val_idx),
        "n_test": len(test_dataset),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "uncertainty_auroc": float(test_uauroc),
        "last_model_path": last_path,
        "best_epoch": tracker.best_epoch,
        "best_val_acc": float(tracker.best_score) if tracker.best_state is not None else None,
        "best_test_loss": float(best_test_loss) if best_test_loss is not None else None,
        "best_test_acc": float(best_test_acc) if best_test_acc is not None else None,
        "best_model_path": best_path,
        "elapsed_sec": elapsed,
    }
    if verbose:
        print(f"[{model_name}/{args.data_name}] ====> acc: {test_acc:.4f} "
              f"(best_epoch {tracker.best_epoch}, best val_acc {tracker.best_score:.4f}, "
              f"best test_acc {best_test_acc if best_test_acc is not None else float('nan'):.4f}) "
              f"(elapsed {elapsed:.1f}s)")
    return model, metrics
