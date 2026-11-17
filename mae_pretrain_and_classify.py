import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import *
import time, copy
import numpy as np
import torch
import torch.nn.functional as F

if len(sys.argv) < 2:
    print("Usage: python mae_pretrain_and_classify.py <2|3>")
    sys.exit(1)

SETTING_IDX = int(sys.argv[1])
ALL_SUBJECTS = list(range(1, 11))

DATA_ROOT       = os.environ.get('DATA_ROOT', os.path.expanduser('~/Desktop/ninapro'))
N_TR_SESSIONS   = 7
TRAIN_PROP      = 0.6
DS              = 30
POST_DS_WINDOW  = 500
PATCH_SIZE      = 20
N_PATCHES       = POST_DS_WINDOW // PATCH_SIZE
EMBED_DIM       = 280
ENC_DEPTH       = 4
NUM_HEADS       = 8
DEC_DIM         = 140
DEC_DEPTH       = 2
DEC_HEADS       = 4
MASK_RATIO      = 0.5
MLP_RATIO       = 4.0
TCN_DEPTH       = 3
TCN_KERNEL      = 4
TCN_DILATIONS   = [2**i for i in range(TCN_DEPTH)]
TCN_RF          = 1 + sum((TCN_KERNEL-1)*d for d in TCN_DILATIONS)
VOTE_START      = TCN_RF - 1
TCN_CHANNELS    = [max(16, 128 // (2**i)) for i in range(TCN_DEPTH)]
DROPOUT         = 0.3
AUX_MODE        = 'supervised'
SUPERVISED_FRAC = 0.05
LAMBDA_AUX      = 0.1
LR_PRETRAIN     = 1.5e-4
LR_FINETUNE     = 3e-4
WEIGHT_DECAY    = 0.05
BATCH_SIZE      = 64
N_PRETRAIN_ITERS = 100_000
N_FINETUNE_ITERS = 30_000
LOG_BASE        = 2
EVAL_INCLUDE_REST = True
SEED = int(os.environ.get('SEED', 42))
INTER_TOLERANCE = float(os.environ.get('INTER_TOLERANCE', 0.02))

SETTINGS_LIST = [
    (True,  True),
    (True,  False),
    (False, True),
    (False, False),
]
SETTING_NAMES = ['mae_aux_frozen', 'mae_aux_unfrozen', 'mae_only_frozen', 'mae_only_unfrozen']

use_aux_loss, freeze_encoder = SETTINGS_LIST[SETTING_IDX]
SETTING_NAME = SETTING_NAMES[SETTING_IDX]
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'mae_cross', f'seed_{SEED}', f'setting_{SETTING_IDX}_{SETTING_NAME}')
os.makedirs(SAVE_DIR, exist_ok=True)

seed_everything(SEED)

print(f'=== MAE Cross-Subject HPC ===')
print(f'Setting index   : {SETTING_IDX} ({SETTING_NAME})')

print('=== Scanning alive channels ===')
alive_mask = scan_common_alive(ALL_SUBJECTS, DATA_ROOT, N_TR_SESSIONS)
IN_CH = int(alive_mask.sum().item())
print(f'Common alive channels: {IN_CH}/16')

per_subject      = {}
pretrain_signals = []
pretrain_labels  = []
for subj in ALL_SUBJECTS:
    d = load_subject_data(subj, DATA_ROOT, DS, N_TR_SESSIONS,
                          TRAIN_PROP, POST_DS_WINDOW, alive_mask)
    if d is None:
        print(f'  S{subj}: no usable data, skipping'); continue
    per_subject[subj] = d
    pretrain_signals.extend(d['pretrain_sigs'])
    pretrain_labels.extend(d['pretrain_lbls'])
print(f'Subjects loaded: {len(per_subject)}, pretrain sessions: {len(pretrain_signals)}')

def pretrain_mae(pretrain_signals, pretrain_labels, use_aux_loss):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mae_model = MAEModel(IN_CH, POST_DS_WINDOW, PATCH_SIZE, EMBED_DIM, ENC_DEPTH, NUM_HEADS,
                         DEC_DIM, DEC_DEPTH, DEC_HEADS, MLP_RATIO, MASK_RATIO).to(device)

    tcn_pretrain = None
    if use_aux_loss:
        tcn_pretrain = TCNHead(EMBED_DIM, N_CLASSES, TCN_DILATIONS, TCN_CHANNELS, TCN_KERNEL, DROPOUT).to(device)

    sup_signals, sup_labels = [], []
    if use_aux_loss:
        sup_n = max(1, int(round(SUPERVISED_FRAC * len(pretrain_signals))))
        rng_idx = np.random.default_rng(SEED).choice(len(pretrain_signals), sup_n, replace=False)
        sup_signals = [pretrain_signals[i] for i in rng_idx]
        sup_labels  = [pretrain_labels[i]  for i in rng_idx]

    params = [p for p in mae_model.parameters() if p.requires_grad]
    if use_aux_loss:
        params += list(tcn_pretrain.parameters())

    optimizer = torch.optim.AdamW(params, lr=LR_PRETRAIN, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_PRETRAIN_ITERS, eta_min=LR_PRETRAIN/10)

    ckpt_set = set(make_log_checkpoints(N_PRETRAIN_ITERS, LOG_BASE))

    run_loss = run_mae = run_aux = run_n = 0.0
    best_loss = float('inf')
    t0 = time.time()

    mae_model.train()
    if use_aux_loss: tcn_pretrain.train()

    hist_iter, hist_mae, hist_aux = [], [], []

    for it in range(1, N_PRETRAIN_ITERS + 1):
        windows, _ = sample_batch(pretrain_signals, pretrain_labels, BATCH_SIZE, POST_DS_WINDOW, PATCH_SIZE, N_CLASSES, device, ds=DS)


        mae_loss = mae_model(windows)
        loss = mae_loss
        m_val = mae_loss.item()
        a_val = 0.0

        if use_aux_loss:
            sup_windows, sup_plabels = sample_batch(sup_signals, sup_labels, BATCH_SIZE, POST_DS_WINDOW, PATCH_SIZE, N_CLASSES, device, ds=DS)
            patch_feats = mae_model.encode_patches(sup_windows)
            aux_loss = compute_aux_loss(AUX_MODE, tcn_pretrain, patch_feats, sup_plabels)
            loss = loss + LAMBDA_AUX * aux_loss
            a_val = aux_loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        run_loss += loss.item()
        run_mae += m_val
        run_aux += a_val
        run_n += 1

        if it in ckpt_set:
            avg = run_loss / run_n
            avg_m = run_mae / run_n
            avg_a = run_aux / run_n
            hist_iter.append(it)
            hist_mae.append(avg_m)
            hist_aux.append(avg_a)

            print(f'  [iter {it}/{N_PRETRAIN_ITERS}] loss={avg:.4f} mae={avg_m:.4f} aux={avg_a:.4f} elapsed={time.time()-t0:.0f}s')
            if avg < best_loss:
                best_loss = avg
                torch.save(mae_model.state_dict(), os.path.join(SAVE_DIR, 'pretrained_best.pt'))

            run_loss = run_mae = run_aux = run_n = 0.0

    torch.save(mae_model.state_dict(), os.path.join(SAVE_DIR, 'pretrained_final.pt'))
    return mae_model, hist_iter, hist_mae, hist_aux

def finetune_subject(mae_model, subject_idx, tr_sigs, tr_lbls, te_i_sigs, te_i_lbls, te_x_sigs, te_x_lbls, freeze_encoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tcn = TCNHead(EMBED_DIM, N_CLASSES, TCN_DILATIONS, TCN_CHANNELS, TCN_KERNEL, DROPOUT).to(device)
    encoder = copy.deepcopy(mae_model).to(device)

    if freeze_encoder:
        for p in encoder.parameters(): p.requires_grad_(False)
        trainable = list(tcn.parameters())
    else:
        trainable = [p for p in encoder.parameters() if p.requires_grad] + list(tcn.parameters())

    optimizer = torch.optim.AdamW(trainable, lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_FINETUNE_ITERS, eta_min=LR_FINETUNE/10)

    ckpt_set = set(make_log_checkpoints(N_FINETUNE_ITERS, LOG_BASE))


    print(f'    [S{subject_idx}] Building gesture index...')
    gesture_idx = build_gesture_index(tr_sigs, tr_lbls, POST_DS_WINDOW, PATCH_SIZE, ds=DS, vote_start=VOTE_START)
    print(f'    [S{subject_idx}] Gesture windows: {len(gesture_idx)}')


    cw = torch.ones(N_CLASSES, device=device)
    cw[0] = 0.5

    hist_iter, hist_intra, hist_inter = [], [], []
    checkpoint_log = []
    snapshots = {}
    run_loss = run_n = 0.0

    encoder.eval() if freeze_encoder else encoder.train()
    tcn.train()

    for it in range(1, N_FINETUNE_ITERS + 1):
        windows, plabels = finetune_sample_batch(
            tr_sigs, tr_lbls, gesture_idx, BATCH_SIZE,
            POST_DS_WINDOW, PATCH_SIZE, N_CLASSES, device, ds=DS)
        feats = encoder.encode_patches(windows)
        if freeze_encoder: feats = feats.detach()
        logits = tcn(feats)

        loss = F.cross_entropy(
            logits.reshape(-1, N_CLASSES),
            plabels.reshape(-1).clamp(0, N_CLASSES - 1),
            weight=cw)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        run_loss += loss.item()
        run_n += 1

        if it in ckpt_set:
            avg = run_loss / run_n
            acc_i = evaluate_sliding_window(encoder, tcn, te_i_sigs, te_i_lbls, POST_DS_WINDOW, PATCH_SIZE, device, VOTE_START, EVAL_INCLUDE_REST)
            acc_x = evaluate_sliding_window(encoder, tcn, te_x_sigs, te_x_lbls, POST_DS_WINDOW, PATCH_SIZE, device, VOTE_START, EVAL_INCLUDE_REST)
            hist_iter.append(it)
            hist_intra.append(acc_i)
            hist_inter.append(acc_x)

            print(f'    [S{subject_idx}|iter {it}] loss={avg:.4f} intra={acc_i:.4f} inter={acc_x:.4f}')
            checkpoint_log.append(dict(it=it, intra_acc=acc_i, inter_acc=acc_x))
            snapshots[it] = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(tcn.state_dict()))

            run_loss = run_n = 0.0
            encoder.eval() if freeze_encoder else encoder.train()
            tcn.train()

    best = select_best_checkpoint(checkpoint_log, INTER_TOLERANCE)
    best_encoder_sd, best_tcn_sd = snapshots[best['it']]
    torch.save({'encoder': best_encoder_sd, 'tcn': best_tcn_sd}, os.path.join(SAVE_DIR, f'finetune_best_S{subject_idx}.pt'))
    torch.save({'encoder': encoder.state_dict(), 'tcn': tcn.state_dict()}, os.path.join(SAVE_DIR, f'finetune_final_S{subject_idx}.pt'))
    print(f"    [S{subject_idx}] Best checkpoint: iter={best['it']} intra={best['intra_acc']:.4f} inter={best['inter_acc']:.4f} "
          f'(selected by best inter_acc, tie-broken by intra_acc within {INTER_TOLERANCE:.3f} tolerance)')
    return hist_iter, hist_intra, hist_inter, best['intra_acc']


if __name__ == '__main__':


    _ckpt_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', f'seed_{SEED}', 'mae_pretrained.pt'),
        os.path.join(SAVE_DIR, 'pretrained_final.pt'),
    ]
    _ckpt_path = next((p for p in _ckpt_candidates if os.path.exists(os.path.abspath(p))), None)

    if _ckpt_path:
        print(f'(pretrain checkpoint already exists at {os.path.abspath(_ckpt_path)}, skipping pretrain)')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mae_model = MAEModel(IN_CH, POST_DS_WINDOW, PATCH_SIZE, EMBED_DIM, ENC_DEPTH, NUM_HEADS,
                             DEC_DIM, DEC_DEPTH, DEC_HEADS, MLP_RATIO, MASK_RATIO)
        state = torch.load(_ckpt_path, map_location=device, weights_only=False)
        mae_model.load_state_dict(state)
        mae_model = mae_model.to(device)
        pt_iters, pt_mae, pt_aux = [], [], []
    else:
        print(f'=== no MAE pretrain checkpoint found -- pretraining from scratch ({N_PRETRAIN_ITERS} iters) ===')
        mae_model, pt_iters, pt_mae, pt_aux = pretrain_mae(pretrain_signals, pretrain_labels, use_aux_loss)

    ft_iters_dict = {}
    ft_intra_dict = {}
    ft_inter_dict = {}
    final_intra = {}
    final_inter = {}

    for subj in ALL_SUBJECTS:
        if subj not in per_subject: continue
        d = per_subject[subj]
        subj_ckpt_path = os.path.join(SAVE_DIR, f'finetune_best_S{subj}.pt')
        if os.path.exists(subj_ckpt_path):


            print(f'  S{subj}: finetune checkpoint already exists, skipping (resume)')
            ft_iters_dict[subj] = []
            ft_intra_dict[subj] = []
            ft_inter_dict[subj] = []
        else:
            ft_iters, ft_intra, ft_inter, best_intra = finetune_subject(mae_model, subj, d['tr_sigs'], d['tr_lbls'], d['te_i_sigs'], d['te_i_lbls'], d['te_x_sigs'], d['te_x_lbls'], freeze_encoder)
            ft_iters_dict[subj] = ft_iters
            ft_intra_dict[subj] = ft_intra
            ft_inter_dict[subj] = ft_inter

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        best_ckpt = torch.load(subj_ckpt_path, weights_only=True)
        encoder = copy.deepcopy(mae_model).to(device)
        encoder.load_state_dict(best_ckpt['encoder'])
        tcn = TCNHead(EMBED_DIM, N_CLASSES, TCN_DILATIONS, TCN_CHANNELS, TCN_KERNEL, DROPOUT).to(device)
        tcn.load_state_dict(best_ckpt['tcn'])

        final_intra[subj] = evaluate_sliding_window(encoder, tcn, d['te_i_sigs'], d['te_i_lbls'], POST_DS_WINDOW, PATCH_SIZE, device, VOTE_START, EVAL_INCLUDE_REST)
        final_inter[subj] = evaluate_sliding_window(encoder, tcn, d['te_x_sigs'], d['te_x_lbls'], POST_DS_WINDOW, PATCH_SIZE, device, VOTE_START, EVAL_INCLUDE_REST)
        print(f'  S{subj} Final Best Intra: {final_intra[subj]:.4f} | Inter: {final_inter[subj]:.4f}')

    npz_path = os.path.join(SAVE_DIR, 'results.npz')
    np.savez(npz_path,
        setting_idx=SETTING_IDX,
        use_aux_loss=use_aux_loss,
        aux_mode=AUX_MODE,
        freeze_encoder=freeze_encoder,
        supervised_frac=SUPERVISED_FRAC,
        lambda_aux=LAMBDA_AUX,
        subjects=ALL_SUBJECTS,
        pretrain_iters=pt_iters,
        pretrain_mae_loss=pt_mae,
        pretrain_aux_loss=pt_aux,
        finetune_iters=ft_iters_dict,
        finetune_acc_intra=ft_intra_dict,
        finetune_acc_inter=ft_inter_dict,
        final_intra=final_intra,
        final_inter=final_inter
    )
    print('Done. NPZ saved to', npz_path)
