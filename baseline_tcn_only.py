import os, sys, copy, warnings, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

from models import (
    N_CLASSES,
    TCNHead,
    build_gesture_index,
    evaluate_sliding_window,
    finetune_sample_batch,
    load_subject_data,
    make_log_checkpoints,
    scan_common_alive,
    seed_everything,
    select_best_checkpoint,
)


print('=== TCN-Only Baseline HPC (no transformer/attention, per-subject) ===')

ALL_SUBJECTS   = list(range(1, 11))
DATA_ROOT      = os.environ.get('DATA_ROOT', os.path.expanduser('~/Desktop/ninapro'))
N_TR_SESSIONS  = 7
TRAIN_PROP     = 0.6
DS             = 30

POST_DS_WINDOW = 500
PATCH_SIZE     = 20
N_PATCHES      = POST_DS_WINDOW // PATCH_SIZE
EMBED_DIM      = 280

TCN_DEPTH      = 3
TCN_KERNEL     = 4
TCN_DILATIONS  = [2**i for i in range(TCN_DEPTH)]
TCN_CHANNELS   = [max(16, 128 // (2**i)) for i in range(TCN_DEPTH)]
TCN_RF         = 1 + sum((TCN_KERNEL - 1) * d for d in TCN_DILATIONS)
VOTE_START     = TCN_RF - 1
EVAL_INCLUDE_REST = True
INTER_TOLERANCE = float(os.environ.get('INTER_TOLERANCE', 0.02))
DROPOUT        = 0.3

N_TRAIN_ITERS  = int(os.environ.get('N_TRAIN_ITERS', 30_000))
LOG_BASE       = 2
BATCH_SIZE     = 64
LR             = 3e-4
WEIGHT_DECAY   = 0.05
SEED = int(os.environ.get('SEED', 42))

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'baseline_emg', f'seed_{SEED}')
os.makedirs(SAVE_DIR, exist_ok=True)
seed_everything(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device : {DEVICE}')


print('\n=== Scanning alive channels ===')
alive_mask = scan_common_alive(ALL_SUBJECTS, DATA_ROOT, N_TR_SESSIONS)
IN_CH = int(alive_mask.sum().item())
print(f'Common alive channels: {IN_CH}/16')
assert IN_CH > 0

print('\n=== Loading subjects ===')
per_subject = {}
for subj in ALL_SUBJECTS:
    d = load_subject_data(subj, DATA_ROOT, DS, N_TR_SESSIONS,
                          TRAIN_PROP, POST_DS_WINDOW, alive_mask)
    if d is None:
        continue
    per_subject[subj] = d
    print(f'  S{subj}: train={len(d["tr_sigs"])}  intra={len(d["te_i_sigs"])}  inter={len(d["te_x_sigs"])}')

class_weights = torch.ones(N_CLASSES)
class_weights[0] = 0.5
class_weights = class_weights.to(DEVICE)

EFFECTIVE_LEN = N_PATCHES * PATCH_SIZE


def patchify(x):


    x = x[:, :, :EFFECTIVE_LEN]
    b, c, t = x.shape
    p = PATCH_SIZE
    return x.reshape(b, c, t // p, p).permute(0, 2, 1, 3).reshape(b, t // p, c * p)


class _EncoderAdapter:


    def __init__(self, embed):
        self.embed = embed

    def encode_patches(self, windows):
        return self.embed(patchify(windows))

    def train(self, mode=True):
        self.embed.train(mode)

    def eval(self):
        self.embed.eval()

    @property
    def training(self):
        return self.embed.training


def train_one_subject(subj, d):
    ckpt_path = os.path.join(SAVE_DIR, f'finetune_best_S{subj}.pt')

    gesture_idx = build_gesture_index(d['tr_sigs'], d['tr_lbls'], POST_DS_WINDOW, PATCH_SIZE, ds=DS, vote_start=VOTE_START)
    print(f'  [S{subj}] gesture-anchored windows: {len(gesture_idx)}')

    patch_embed = nn.Linear(IN_CH * PATCH_SIZE, EMBED_DIM).to(DEVICE)
    tcn = TCNHead(EMBED_DIM, N_CLASSES, TCN_DILATIONS, TCN_CHANNELS, TCN_KERNEL, DROPOUT).to(DEVICE)
    encoder = _EncoderAdapter(patch_embed)

    if os.path.exists(ckpt_path):
        print(f'  [S{subj}] checkpoint already exists, skipping training (resume)')
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        patch_embed.load_state_dict(state['patch_embed'])
        tcn.load_state_dict(state['tcn'])
        acc_i = evaluate_sliding_window(encoder, tcn, d['te_i_sigs'], d['te_i_lbls'],
                                        POST_DS_WINDOW, PATCH_SIZE, DEVICE, VOTE_START, EVAL_INCLUDE_REST)
        acc_x = evaluate_sliding_window(encoder, tcn, d['te_x_sigs'], d['te_x_lbls'],
                                        POST_DS_WINDOW, PATCH_SIZE, DEVICE, VOTE_START, EVAL_INCLUDE_REST)
        return acc_i, acc_x

    params = list(patch_embed.parameters()) + list(tcn.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_TRAIN_ITERS, eta_min=LR / 10)
    ckpt_set = set(make_log_checkpoints(N_TRAIN_ITERS, LOG_BASE))

    run_loss = run_n = 0.0
    checkpoint_log = []
    snapshots = {}
    t0 = time.time()

    patch_embed.train(); tcn.train()
    for it in range(1, N_TRAIN_ITERS + 1):
        windows, plabels = finetune_sample_batch(
            d['tr_sigs'], d['tr_lbls'], gesture_idx, BATCH_SIZE,
            POST_DS_WINDOW, PATCH_SIZE, N_CLASSES, DEVICE, ds=DS)

        feats = patch_embed(patchify(windows))
        logits = tcn(feats)
        loss = F.cross_entropy(logits.reshape(-1, N_CLASSES), plabels.reshape(-1).clamp(0, N_CLASSES - 1), weight=class_weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        run_loss += loss.item()
        run_n += 1

        if it in ckpt_set:
            avg = run_loss / run_n
            acc_i = evaluate_sliding_window(encoder, tcn, d['te_i_sigs'], d['te_i_lbls'],
                                            POST_DS_WINDOW, PATCH_SIZE, DEVICE, VOTE_START, EVAL_INCLUDE_REST)
            acc_x = evaluate_sliding_window(encoder, tcn, d['te_x_sigs'], d['te_x_lbls'],
                                            POST_DS_WINDOW, PATCH_SIZE, DEVICE, VOTE_START, EVAL_INCLUDE_REST)
            print(f'    [S{subj}|iter {it:6d}/{N_TRAIN_ITERS}] loss={avg:.4f} intra={acc_i:.4f} inter={acc_x:.4f} ({time.time()-t0:.0f}s)')
            checkpoint_log.append(dict(it=it, intra_acc=acc_i, inter_acc=acc_x))
            snapshots[it] = (copy.deepcopy(patch_embed.state_dict()), copy.deepcopy(tcn.state_dict()))
            run_loss = run_n = 0.0
            patch_embed.train(); tcn.train()

    best = select_best_checkpoint(checkpoint_log, INTER_TOLERANCE)
    best_pe_sd, best_tcn_sd = snapshots[best['it']]
    torch.save({'patch_embed': best_pe_sd, 'tcn': best_tcn_sd}, ckpt_path)
    torch.save({'patch_embed': patch_embed.state_dict(), 'tcn': tcn.state_dict()},
               os.path.join(SAVE_DIR, f'finetune_final_S{subj}.pt'))
    print(f"  [S{subj}] BEST iter={best['it']} intra={best['intra_acc']:.4f} inter={best['inter_acc']:.4f} "
          f'(selected by best inter_acc, tie-broken by intra_acc within {INTER_TOLERANCE:.3f} tolerance)')
    return best['intra_acc'], best['inter_acc']


if __name__ == '__main__':
    final_intra, final_inter = {}, {}
    for subj in ALL_SUBJECTS:
        if subj not in per_subject:
            continue
        acc_i, acc_x = train_one_subject(subj, per_subject[subj])
        final_intra[subj] = acc_i
        final_inter[subj] = acc_x

    mean_i = float(np.mean(list(final_intra.values())))
    mean_x = float(np.mean(list(final_inter.values())))
    print(f'\n=== Done. Per-subject mean intra={mean_i:.4f} inter={mean_x:.4f} ({len(final_intra)} subjects) ===')

    np.savez(os.path.join(SAVE_DIR, 'results.npz'),
        subjects=np.array(sorted(final_intra.keys())),
        final_intra=final_intra,
        final_inter=final_inter,
        mean_intra=mean_i,
        mean_inter=mean_x,
    )
    print('Results saved to results.npz')
