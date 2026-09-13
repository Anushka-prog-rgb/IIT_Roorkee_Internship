import contextlib
import copy
import io
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm
from model import TMC
from data import Multi_view_data
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BestTracker:
    """
    Tracks the best validation score seen so far (optionally across several
    sequential training stages sharing one tracker) and a snapshot of the
    given module's weights at that point. Unlike the earlier best-epoch
    mechanism this drives no decisions -- no early stopping, no restoring
    into the model that gets evaluated -- it exists purely so the
    best-scoring checkpoint can be saved to disk alongside the final-epoch
    one for later comparison.
    """
    def __init__(self):
        self.best_score = -float('inf')
        self.best_epoch = None
        self.best_state = None

    def step(self, score, module, epoch):
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(module.state_dict())
            return True
        return False


def split_val_indices(dataset, val_size=0.15, random_state=0):
    """
    Stratified split of an already train/test-partitioned dataset into a
    core-training and validation partition. The validation slice is never
    trained on -- it exists solely to score epochs so the best-scoring
    checkpoint can be identified and saved, distinct from the final-epoch
    checkpoint.
    """
    n = len(dataset)
    idx = np.arange(n)
    labels = dataset.y
    try:
        train_idx, val_idx = train_test_split(
            idx, test_size=val_size, random_state=random_state, stratify=labels)
    except ValueError:
        # some classes too small to stratify a further split off of -- fall
        # back to a plain random split rather than failing the whole run
        train_idx, val_idx = train_test_split(
            idx, test_size=val_size, random_state=random_state)
    return train_idx, val_idx


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, optimizer, loader, epoch, use_reliability, device=DEVICE, clip_grad_norm=5.0):
    model.train()
    loss_meter = AverageMeter()
    for batch_idx, (data, target) in enumerate(loader):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].to(device))
        target = Variable(target.long().to(device))
        optimizer.zero_grad()
        evidences, evidence_a, loss, _ = model(data, target, epoch, use_reliability=use_reliability)
        loss.backward()
        if clip_grad_norm is not None:
            # the KL term's annealing coefficient can produce sharp gradient
            # spikes early in training (small global_step / annealing_step),
            # clipping keeps a single bad batch from blowing up the Adam
            # moment estimates for the rest of the run
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()
        loss_meter.update(loss.item())
    return loss_meter.avg


def eval_epoch(model, loader, epoch, use_reliability, device=DEVICE):
    model.eval()
    loss_meter = AverageMeter()
    correct_num, data_num = 0, 0
    for batch_idx, (data, target) in enumerate(loader):
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].to(device))
        data_num += target.size(0)
        with torch.no_grad():
            target = Variable(target.long().to(device))
            evidences, evidence_a, loss, _ = model(data, target, epoch, use_reliability=use_reliability)
            _, predicted = torch.max(evidence_a.data, 1)
            correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())
    return loss_meter.avg, correct_num / data_num


def compute_oof_reliability_targets(args, train_dataset, device=DEVICE, progress=True):
    """
    Path 1 -- learn reliability against outcomes, cross-sample.

    Reliability of view m can only be estimated against ground truth over a
    population, not from a single prediction, and it must not be read off
    the same fit that produced the prediction (that's in-sample-optimistic
    and just re-derives the view's own confidence). So: split the training
    set into K folds, fit a classifier on K-1 folds, and record whether each
    view is actually correct on the held-out fold. Every training sample
    ends up with an out-of-fold correctness label per view -- this is the
    ground-truth signal r_m(x) is later trained against.
    """
    n = len(train_dataset)
    oof_correct = {v: np.zeros(n, dtype=np.float32) for v in range(args.views)}
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=0)
    indices = np.arange(n)

    fold_enum = enumerate(kf.split(indices))
    if progress:
        fold_enum = tqdm(fold_enum, total=args.k_folds, desc='cross-fit folds', unit='fold', leave=False)
    for fold_idx, (fit_idx, holdout_idx) in fold_enum:
        print('--- cross-fitting fold %d/%d (fit=%d, holdout=%d) ---' %
              (fold_idx + 1, args.k_folds, len(fit_idx), len(holdout_idx)))

        fit_loader = DataLoader(Subset(train_dataset, fit_idx), batch_size=args.batch_size, shuffle=True)
        holdout_loader = DataLoader(Subset(train_dataset, holdout_idx), batch_size=args.batch_size, shuffle=False)

        fold_model = TMC(args.classes, args.views, args.dims, args.lambda_epochs).to(device)
        fold_optimizer = optim.Adam(fold_model.parameters(), lr=args.lr, weight_decay=1e-5)

        epoch_bar = range(1, args.fold_epochs + 1)
        if progress:
            epoch_bar = tqdm(epoch_bar, desc='  fold %d/%d epochs' % (fold_idx + 1, args.k_folds),
                              unit='epoch', leave=False)
        for epoch in epoch_bar:
            # use_reliability=False: the fold model only needs to produce a
            # view-wise classifier here, the reliability heads themselves
            # aren't the object of cross-fitting, the classifiers are
            train_loss = train_epoch(fold_model, fold_optimizer, fit_loader, epoch, use_reliability=False, device=device)
            if progress:
                epoch_bar.set_postfix(loss='%.3f' % train_loss)

        fold_model.eval()
        pos = 0
        with torch.no_grad():
            for data, target in holdout_loader:
                bs = target.size(0)
                for v_num in range(len(data)):
                    data[v_num] = Variable(data[v_num].to(device))
                target = Variable(target.long().to(device))
                evidence = fold_model.infer(data)
                global_idx = holdout_idx[pos:pos + bs]
                for v_num in range(args.views):
                    pred = torch.argmax(evidence[v_num], dim=1)
                    correct = (pred == target).float().cpu().numpy()
                    oof_correct[v_num][global_idx] = correct
                pos += bs

    for v_num in range(args.views):
        print('view %d out-of-fold accuracy (reliability base rate): %.4f' %
              (v_num, oof_correct[v_num].mean()))
    return oof_correct


def train_reliability_heads(args, model, train_dataset, oof_correct, device=DEVICE, progress=True):
    """
    Fit each view's ReliabilityNet to predict its own out-of-fold
    correctness from x. Because the targets came from held-out folds, the
    resulting r_m(x) reflects where view m is actually right in this region
    of input space, not how certain view m's own evidence network claims to
    be -- that's precisely what lets a confidently-wrong view get demoted.
    """
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    targets = {v: torch.from_numpy(oof_correct[v]).to(device) for v in range(args.views)}

    params = []
    for v in range(args.views):
        params += list(model.Reliability[v].parameters())
    rel_optimizer = optim.Adam(params, lr=args.lr)

    epoch_bar = range(1, args.reliability_epochs + 1)
    if progress:
        epoch_bar = tqdm(epoch_bar, desc='reliability heads', unit='epoch', leave=False)
    for epoch in epoch_bar:
        meter = AverageMeter()
        pos = 0
        for data, target in loader:
            bs = target.size(0)
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].to(device))
            rel_optimizer.zero_grad()
            reliability = model.reliability(data)
            rel_loss = 0
            for v_num in range(args.views):
                pred_r = reliability[v_num].squeeze(-1)
                tgt_r = targets[v_num][pos:pos + bs]
                rel_loss += F.binary_cross_entropy(pred_r, tgt_r)
            rel_loss.backward()
            rel_optimizer.step()
            meter.update(rel_loss.item())
            pos += bs
        if progress:
            epoch_bar.set_postfix(bce_loss='%.3f' % meter.avg)
        elif epoch % 10 == 0 or epoch == args.reliability_epochs:
            print('reliability head epoch %d, bce loss %.4f' % (epoch, meter.avg))


def load_datasets(args):
    """
    Loads train/test datasets for args.data_path and fills in args.dims /
    args.views / args.classes from the loaded shapes -- these vary per
    dataset (handwritten_6views: 6 views / 10 classes, scene15: 3 views /
    15 classes, Cora: 4 views / 7 classes, ...) so they're read off the data
    instead of hardcoded.
    """
    train_dataset = Multi_view_data(args.data_path, train=True)
    test_dataset = Multi_view_data(args.data_path, train=False)
    args.dims = [[train_dataset.X[v].shape[1]] for v in range(len(train_dataset.X))]
    args.views = len(args.dims)
    args.classes = int(max(train_dataset.y.max(), test_dataset.y.max())) + 1
    return train_dataset, test_dataset


def list_dataset_names(datasets_dir='datasets'):
    """Base names (no .mat extension) of every dataset file under datasets_dir, sorted."""
    return sorted(
        os.path.splitext(f)[0] for f in os.listdir(datasets_dir) if f.endswith('.mat')
    )


def save_checkpoint(state_dict, checkpoint_dir, filename):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state_dict, path)
    return path


def run_pipeline(args, device=DEVICE, verbose=True, progress=True):
    """
    Runs the full 3-stage TMC pipeline (cross-fitting -> reliability heads ->
    gated final training) against whatever dataset args.data_path points at.
    Every stage trains for its full fixed epoch budget; the model used for
    the main reported evaluation is whatever the last epoch produced -- no
    validation-based early stopping or restoring.

    Stage 3 additionally holds out a stratified slice of the training set
    (args.val_size, default 15%, never trained on) purely to score each
    epoch, so the best-scoring checkpoint can be saved to disk alongside the
    final-epoch one under args.checkpoint_dir (default 'checkpoints') --
    two separate .pt files, for later comparison. This doesn't change which
    model is reported as 'the' result (still the final epoch, matching
    train_loss/test_acc above); 'best_*' metrics are informational.

    Returns (model, metrics) where metrics is a dict covering dataset shape,
    per-view out-of-fold reliability, final and best losses/accuracy,
    checkpoint paths, and wall-clock time.
    """
    t0 = time.time()
    train_dataset, test_dataset = load_datasets(args)
    val_size = getattr(args, 'val_size', 0.15)
    train_core_idx, val_idx = split_val_indices(train_dataset, val_size=val_size)
    train_loader = DataLoader(Subset(train_dataset, train_core_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(train_dataset, val_idx), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    if verbose:
        print('[%s] views=%d, dims=%s, classes=%d, train=%d (core=%d, val=%d), test=%d' % (
            args.data_name, args.views, args.dims, args.classes,
            len(train_dataset), len(train_core_idx), len(val_idx), len(test_dataset)))

    # compute_oof_reliability_targets/train_reliability_heads print per-fold
    # and per-epoch progress unconditionally; route that to nowhere when
    # verbose=False so a bulk sweep over many datasets doesn't get spammed.
    # tqdm writes to stderr (or renders as an ipywidget in a notebook), so
    # progress bars stay visible either way.
    sink = contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext()

    stage_bar = tqdm(total=3, desc='[%s] pipeline' % args.data_name, unit='stage') if progress else None

    # --- Path 1: learn reliability against outcomes, cross-sample ---
    # Stage 1: cross-fit per-view classifiers across K folds to obtain
    # held-out correctness labels for every training sample and view. Runs
    # over the *full* train_dataset (fold models are discarded after use,
    # so this can't leak into the final model or its checkpoint selection).
    if stage_bar is not None:
        stage_bar.set_description('[%s] cross-fitting' % args.data_name)
    with sink:
        oof_correct = compute_oof_reliability_targets(args, train_dataset, device=device, progress=progress)
    if stage_bar is not None:
        stage_bar.update(1)

    # Stage 2: fit the reliability heads r_m(x) against those out-of-fold
    # labels on a fresh model instance, over the full train_dataset.
    model = TMC(args.classes, args.views, args.dims, args.lambda_epochs)
    model.to(device)
    if stage_bar is not None:
        stage_bar.set_description('[%s] reliability heads' % args.data_name)
    with sink:
        train_reliability_heads(args, model, train_dataset, oof_correct, device=device, progress=progress)
    if stage_bar is not None:
        stage_bar.update(1)

    # Freeze the reliability heads before training the final classifiers so
    # they stay anchored to the cross-fitted targets instead of co-adapting
    # in-sample with the classifiers they are meant to be gating.
    for v in range(args.views):
        for p in model.Reliability[v].parameters():
            p.requires_grad = False

    # Stage 3: train the final per-view classifiers on train_core, with
    # fusion gated by the frozen, cross-fitted reliability scores. val_idx
    # never enters a gradient update here -- it only scores each epoch so
    # the best-by-validation snapshot can be kept alongside the last one.
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=getattr(args, 'weight_decay', 1e-5))
    tracker = BestTracker()

    if stage_bar is not None:
        stage_bar.set_description('[%s] final training' % args.data_name)
    epoch_bar = range(1, args.epochs + 1)
    if progress:
        epoch_bar = tqdm(epoch_bar, desc='  final training epochs', unit='epoch', leave=False)
    train_loss = None
    with sink:
        for epoch in epoch_bar:
            train_loss = train_epoch(model, optimizer, train_loader, epoch, use_reliability=True, device=device)
            _, val_acc = eval_epoch(model, val_loader, epoch, use_reliability=True, device=device)
            tracker.step(val_acc, model, epoch)
            if progress:
                epoch_bar.set_postfix(loss='%.3f' % train_loss, val_acc='%.3f' % val_acc,
                                       best='%.3f' % tracker.best_score)
    if stage_bar is not None:
        stage_bar.update(1)
        stage_bar.close()

    # last-epoch model: whatever `model` holds right now (unchanged).
    last_state = copy.deepcopy(model.state_dict())
    test_loss, test_acc = eval_epoch(model, test_loader, args.epochs, use_reliability=True, device=device)

    # briefly swap in the best-by-validation snapshot to get its test
    # accuracy too, then restore last_state so the returned `model` -- and
    # test_loss/test_acc above -- are still the final-epoch model.
    best_test_loss, best_test_acc = None, None
    if tracker.best_state is not None:
        model.load_state_dict(tracker.best_state)
        best_test_loss, best_test_acc = eval_epoch(model, test_loader, tracker.best_epoch,
                                                     use_reliability=True, device=device)
        model.load_state_dict(last_state)

    checkpoint_dir = getattr(args, 'checkpoint_dir', 'checkpoints')
    last_path = save_checkpoint(last_state, checkpoint_dir, '%s_TMC_last.pt' % args.data_name)
    best_path = (save_checkpoint(tracker.best_state, checkpoint_dir, '%s_TMC_best.pt' % args.data_name)
                 if tracker.best_state is not None else None)

    elapsed = time.time() - t0

    metrics = {
        'dataset': args.data_name,
        'views': args.views,
        'classes': args.classes,
        'dims': [d[0] for d in args.dims],
        'n_train': len(train_dataset),
        'n_val': len(val_idx),
        'n_test': len(test_dataset),
        'oof_reliability': {v: float(oof_correct[v].mean()) for v in range(args.views)},
        'train_loss': float(train_loss) if train_loss is not None else None,
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'last_model_path': last_path,
        'best_epoch': tracker.best_epoch,
        'best_val_acc': float(tracker.best_score) if tracker.best_state is not None else None,
        'best_test_loss': float(best_test_loss) if best_test_loss is not None else None,
        'best_test_acc': float(best_test_acc) if best_test_acc is not None else None,
        'best_model_path': best_path,
        'elapsed_sec': elapsed,
    }
    if verbose:
        print('[%s] ====> acc: %.4f (best_epoch %s/%d, best val_acc %.4f, best test_acc %.4f) (elapsed %.1fs)' % (
            args.data_name, test_acc, tracker.best_epoch, args.epochs,
            tracker.best_score, best_test_acc if best_test_acc is not None else float('nan'), elapsed))
    return model, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train the final model [default: 500]')
    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--k-folds', type=int, default=5, metavar='K',
                        help='number of cross-fitting folds used to estimate out-of-fold reliability')
    parser.add_argument('--fold-epochs', type=int, default=100, metavar='N',
                        help='epochs to train each cross-fitting fold classifier')
    parser.add_argument('--reliability-epochs', type=int, default=50, metavar='N',
                        help='epochs to train the reliability heads against out-of-fold targets')
    parser.add_argument('--weight-decay', type=float, default=1e-5, metavar='WD',
                        help='weight decay (L2) for the final classifier optimizer')
    parser.add_argument('--val-size', type=float, default=0.15, metavar='F',
                        help='fraction of the training set held out (stratified) purely to score '
                             'epochs for best-checkpoint saving; never used for gradient updates')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', metavar='DIR',
                        help='directory to save the best-by-validation and last-epoch checkpoints to')
    parser.add_argument('--no-progress', action='store_true',
                        help='disable tqdm progress bars (they go to stderr by default)')
    parser.add_argument('--dataset', type=str, default='scene15', metavar='NAME',
                        help="dataset name (datasets/<name>.mat), or 'all' to run every "
                             "dataset found in datasets/")
    args = parser.parse_args()

    print('Using device: %s' % DEVICE)

    if args.dataset == 'all':
        results = {}
        dataset_bar = tqdm(list_dataset_names(), desc='datasets', unit='dataset') if not args.no_progress else list_dataset_names()
        for name in dataset_bar:
            args.data_name = name
            args.data_path = 'datasets/' + name
            try:
                _, metrics = run_pipeline(args, device=DEVICE, progress=not args.no_progress)
                results[name] = metrics
            except Exception as e:
                print('[%s] FAILED: %s' % (name, e))
                results[name] = None

        print('\n=== summary ===')
        for name, metrics in results.items():
            print('%-20s %s' % (name, 'FAILED' if metrics is None else '%.4f' % metrics['test_acc']))
    else:
        args.data_name = args.dataset
        args.data_path = 'datasets/' + args.data_name
        run_pipeline(args, device=DEVICE, progress=not args.no_progress)
