import os, sys, copy, random, math
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import io
from scipy.signal import decimate as scipy_decimate
from torch.nn.utils.parametrizations import weight_norm

LABEL_REMAP = {0: 0, 1: 1, 3: 2, 4: 3, 6: 4, 9: 5, 10: 6, 11: 7}
N_CLASSES   = 8

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def make_log_checkpoints(n_iters, base=2):
    pts, k = set(), 1
    while k < n_iters:
        pts.add(k)
        k = max(k + 1, int(round(k * base)))
    pts.add(n_iters)
    return sorted(pts)

def select_best_checkpoint(checkpoint_log, inter_tolerance=0.02):


    best_inter = max(c['inter_acc'] for c in checkpoint_log)
    candidates = [c for c in checkpoint_log if c['inter_acc'] >= best_inter - inter_tolerance]
    if all('intra_acc' in c for c in candidates):
        return max(candidates, key=lambda c: (c['intra_acc'], c['inter_acc'], c['it']))
    return max(candidates, key=lambda c: (c['inter_acc'], c['it']))

def get_sincos_pos_embed(embed_dim, length):
    pos   = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    omega = 1.0 / (10000 ** (torch.arange(embed_dim // 2).float() / (embed_dim / 2)))
    out   = pos * omega.unsqueeze(0)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

def build_causal_mask(seq_len, device):
    return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x if self.chomp_size == 0 else x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.3):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
            weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.BatchNorm1d(out_ch), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
        for m in self.net:
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, 0, 0.01)
    def forward(self, x):
        return self.relu(self.net(x) + (x if self.downsample is None else self.downsample(x)))

class TCNHead(nn.Module):
    def __init__(self, embed_dim, n_classes, dilations, channels, kernel_size=4, dropout=0.3):
        super().__init__()
        layers = []
        for i, (d, out_ch) in enumerate(zip(dilations, channels)):
            in_ch = embed_dim if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, d, dropout))
        self.tcn  = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], n_classes)
        self.proj = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]), nn.GELU(),
            nn.Linear(channels[-1], 128))
    def forward_features(self, x):
        return self.tcn(x.transpose(1, 2)).transpose(1, 2)
    def forward(self, x):
        return self.head(self.forward_features(x))
    def forward_contrastive_features(self, x):
        return F.normalize(self.proj(self.forward_features(x)), dim=-1)

class JEPAModel(nn.Module):
    def __init__(self, in_ch, window_len, patch_size, embed_dim, depth, num_heads,
                 pred_dim, pred_depth, pred_heads, mlp_ratio=4.0,
                 mask_ratio=0.5, ema_momentum=0.996, feat_dim=0):
        super().__init__()
        self.patch_size    = patch_size
        self.mask_ratio    = mask_ratio
        self.ema_momentum  = ema_momentum
        self.effective_len = (window_len // patch_size) * patch_size
        self.num_patches   = self.effective_len // patch_size
        patch_dim = patch_size * in_ch

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed   = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks      = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.tgt_patch_embed = nn.Linear(patch_dim, embed_dim)
        self.tgt_blocks      = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True)
            for _ in range(depth)])
        self.tgt_norm = nn.LayerNorm(embed_dim, eps=1e-6)

        self.pred_proj      = nn.Linear(embed_dim, pred_dim)
        self.pred_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, pred_dim), requires_grad=False)
        self.pred_blocks    = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=pred_dim, nhead=pred_heads,
                dim_feedforward=int(pred_dim * mlp_ratio), batch_first=True)
            for _ in range(pred_depth)])
        self.pred_norm = nn.LayerNorm(pred_dim, eps=1e-6)
        self.pred_out  = nn.Linear(pred_dim, embed_dim)

        self.feat_proj = nn.Linear(feat_dim, embed_dim) if feat_dim > 0 else None

        with torch.no_grad():
            self.pos_embed.copy_(
                get_sincos_pos_embed(embed_dim, self.num_patches).unsqueeze(0))
            self.pred_pos_embed.copy_(
                get_sincos_pos_embed(pred_dim, self.num_patches).unsqueeze(0))

        self._init_target_encoder()

    def _ctx_params(self):
        return (list(self.patch_embed.parameters()) +
                list(self.blocks.parameters()) +
                list(self.norm.parameters()))

    def _tgt_params(self):
        return (list(self.tgt_patch_embed.parameters()) +
                list(self.tgt_blocks.parameters()) +
                list(self.tgt_norm.parameters()))

    def _init_target_encoder(self):
        for p_c, p_t in zip(self._ctx_params(), self._tgt_params()):
            p_t.data.copy_(p_c.data)
            p_t.requires_grad = False

    @torch.no_grad()
    def update_ema(self):
        m = self.ema_momentum
        for p_c, p_t in zip(self._ctx_params(), self._tgt_params()):
            p_t.data.mul_(m).add_(p_c.data, alpha=1.0 - m)

    def train(self, mode=True):
        super().train(mode)
        for blk in self.tgt_blocks:
            blk.eval()
        return self

    def patchify(self, x):
        b, c, t = x.shape
        p = self.patch_size
        t = (t // p) * p
        return x[:, :, :t].reshape(b, c, t // p, p).permute(0, 2, 1, 3).reshape(b, t // p, c * p)

    def _encode_context(self, x_patches, ids_keep):
        b     = x_patches.shape[0]
        tokens = self.patch_embed(x_patches) + self.pos_embed
        x_vis  = torch.gather(tokens, 1,
                               ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        x      = torch.cat([self.cls_token.expand(b, -1, -1), x_vis], dim=1)
        mask   = build_causal_mask(x.shape[1], x.device)
        for blk in self.blocks:
            x = blk(x, src_mask=mask)
        return self.norm(x)

    @torch.no_grad()
    def _encode_target_base(self, x_patches):
        b      = x_patches.shape[0]
        tokens = self.tgt_patch_embed(x_patches) + self.pos_embed
        x      = torch.cat([self.cls_token.expand(b, -1, -1).detach(), tokens], dim=1)
        mask   = build_causal_mask(x.shape[1], x.device)
        for blk in self.tgt_blocks:
            x = blk(x, src_mask=mask)
        return self.tgt_norm(x)[:, 1:, :]

    def get_target_latents(self, x_patches, emg_features=None):
        latents = self._encode_target_base(x_patches)
        if emg_features is not None and self.feat_proj is not None:
            latents = latents + self.feat_proj(emg_features)
        return latents

    def _predict(self, ctx_repr, ids_keep, ids_masked, b):
        N = self.num_patches
        d = self.pred_pos_embed.shape[-1]

        ctx_proj  = self.pred_proj(ctx_repr)
        cls_proj  = ctx_proj[:, :1, :]
        vis_proj  = ctx_proj[:, 1:, :]

        query_pos = torch.gather(
            self.pred_pos_embed.expand(b, -1, -1), 1,
            ids_masked.unsqueeze(-1).expand(-1, -1, d))

        x_full = query_pos.new_zeros(b, N, d)
        x_full.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, d), vis_proj)
        x_full.scatter_(1, ids_masked.unsqueeze(-1).expand(-1, -1, d), query_pos)

        x    = torch.cat([cls_proj, x_full], dim=1)
        mask = build_causal_mask(x.shape[1], x.device)
        for blk in self.pred_blocks:
            x = blk(x, src_mask=mask)
        x         = self.pred_norm(x)
        pred_full = self.pred_out(x[:, 1:, :])

        return torch.gather(pred_full, 1,
                             ids_masked.unsqueeze(-1).expand(-1, -1, pred_full.shape[-1]))

    def encode_patches(self, windows):
        x  = windows[:, :, :self.effective_len]
        b  = x.shape[0]
        xp = self.patchify(x)
        ids_keep = torch.arange(self.num_patches, device=x.device).unsqueeze(0).expand(b, -1)
        ctx_repr = self._encode_context(xp, ids_keep)
        return ctx_repr[:, 1:, :]

class MAEModel(nn.Module):
    def __init__(self, in_ch, window_len, patch_size, embed_dim, depth, num_heads,
                 dec_dim, dec_depth, dec_heads, mlp_ratio, mask_ratio):
        super().__init__()
        self.patch_size   = patch_size
        self.mask_ratio   = mask_ratio
        self.num_patches  = window_len // patch_size
        patch_dim         = in_ch * patch_size
        self.effective_len = self.num_patches * patch_size

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed   = nn.Parameter(
            get_sincos_pos_embed(embed_dim, self.num_patches).unsqueeze(0), requires_grad=False)
        self.blocks      = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads,
                                       int(embed_dim * mlp_ratio),
                                       dropout=0.0, batch_first=True)
            for _ in range(depth)])
        self.norm        = nn.LayerNorm(embed_dim)

        self.dec_proj    = nn.Linear(embed_dim, dec_dim)
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos     = nn.Parameter(
            get_sincos_pos_embed(dec_dim, self.num_patches).unsqueeze(0), requires_grad=False)
        self.dec_blocks  = nn.ModuleList([
            nn.TransformerEncoderLayer(dec_dim, dec_heads,
                                       int(dec_dim * mlp_ratio),
                                       dropout=0.0, batch_first=True)
            for _ in range(dec_depth)])
        self.dec_norm    = nn.LayerNorm(dec_dim)
        self.dec_out     = nn.Linear(dec_dim, patch_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, x):
        B, C, T = x.shape
        N, P    = self.num_patches, self.patch_size
        return x[:, :, :N*P].reshape(B, C, N, P).permute(0, 2, 1, 3).reshape(B, N, C*P)

    def encode_patches(self, x):
        xp  = self.patchify(x)
        tok = self.patch_embed(xp) + self.pos_embed
        cls = self.cls_token.expand(tok.shape[0], -1, -1)
        h   = torch.cat([cls, tok], dim=1)
        mask = build_causal_mask(h.shape[1], h.device)
        for blk in self.blocks:
            h = blk(h, src_mask=mask)
        h = self.norm(h)
        return h[:, 1:, :]

    def reconstruct(self, x, ids_k=None, ids_m=None):


        B  = x.shape[0]
        xp = self.patchify(x)
        N  = self.num_patches

        if ids_k is None:
            noise  = torch.rand(B, N, device=x.device)
            ids_sh = torch.argsort(noise, dim=1)
            n_keep = int(N * (1 - self.mask_ratio))
            ids_k  = torch.sort(ids_sh[:, :n_keep], dim=1).values
            ids_m  = torch.sort(ids_sh[:, n_keep:],  dim=1).values

        vis_patches = torch.gather(xp, 1, ids_k.unsqueeze(-1).expand(-1, -1, xp.shape[-1]))
        tok  = self.patch_embed(vis_patches) + torch.gather(
            self.pos_embed.expand(B, -1, -1), 1,
            ids_k.unsqueeze(-1).expand(-1, -1, self.pos_embed.shape[-1]))
        cls  = self.cls_token.expand(B, -1, -1)
        h    = torch.cat([cls, tok], dim=1)
        enc_mask = build_causal_mask(h.shape[1], h.device)
        for blk in self.blocks:
            h = blk(h, src_mask=enc_mask)
        h = self.norm(h)
        enc_vis = h[:, 1:, :]

        E2  = self.dec_proj.out_features

        with torch.no_grad():
            dec_base = self.mask_token.expand(B, N, -1).clone()

        proj_vis = self.dec_proj(enc_vis)
        dec_tokens = dec_base.scatter(
            1,
            ids_k.unsqueeze(-1).expand(-1, -1, E2),
            proj_vis
        )
        dec_tokens = dec_tokens + self.dec_pos.expand(B, -1, -1)
        dec_mask = build_causal_mask(dec_tokens.shape[1], dec_tokens.device)
        for blk in self.dec_blocks:
            dec_tokens = blk(dec_tokens, src_mask=dec_mask)
        dec_tokens = self.dec_norm(dec_tokens)

        full_pred = self.dec_out(dec_tokens)
        return xp, ids_k, ids_m, full_pred

    def forward(self, x):
        xp, ids_k, ids_m, full_pred = self.reconstruct(x)
        pred_masked = torch.gather(full_pred, 1, ids_m.unsqueeze(-1).expand(-1, -1, xp.shape[-1]))
        tgt_masked  = torch.gather(xp, 1, ids_m.unsqueeze(-1).expand(-1, -1, xp.shape[-1]))
        loss = F.mse_loss(pred_masked, tgt_masked)
        return loss


class ViTEncoder(nn.Module):
    def __init__(self, in_ch, window_len, patch_size, embed_dim, depth, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.patch_size   = patch_size
        self.num_patches  = window_len // patch_size
        self.effective_len = self.num_patches * patch_size
        patch_dim = patch_size * in_ch

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.pos_embed   = nn.Parameter(
            get_sincos_pos_embed(embed_dim, self.num_patches).unsqueeze(0), requires_grad=False)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks      = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio), batch_first=True)
            for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def encode_patches(self, windows):
        x  = windows[:, :, :self.effective_len]
        b, c, t = x.shape
        p = self.patch_size
        xp = x.reshape(b, c, t // p, p).permute(0, 2, 1, 3).reshape(b, t // p, c * p)

        tok = self.patch_embed(xp) + self.pos_embed.expand(b, -1, -1)
        cls = self.cls_token.expand(b, -1, -1)
        h = torch.cat([cls, tok], dim=1)

        mask = build_causal_mask(h.shape[1], h.device)
        for blk in self.blocks:
            h = blk(h, src_mask=mask)
        h = self.norm(h)
        return h[:, 1:, :]

def supervised_loss_fn(logits, patch_labels):
    B, N, C = logits.shape
    return F.cross_entropy(logits.reshape(-1, C), patch_labels.reshape(-1).long())

def supcon_loss_fn(z, labels, temperature=0.3):
    n    = z.shape[0]
    sim  = torch.mm(z, z.T) / temperature
    lbls = labels.view(-1, 1)
    pos  = (lbls == lbls.T).float(); pos.fill_diagonal_(0)
    self_ = torch.eye(n, device=z.device).bool()
    log_d = torch.logsumexp(sim.masked_fill(self_, float('-inf')), dim=1, keepdim=True)
    log_p = sim - log_d
    return (-(pos * log_p).sum(dim=1) / pos.sum(dim=1).clamp_min(1)).mean()

def compute_aux_loss(aux_mode, tcn, patch_feats, patch_labels):
    if aux_mode == 'supervised':
        return supervised_loss_fn(tcn(patch_feats), patch_labels)
    elif aux_mode == 'contrastive':
        z = tcn.forward_contrastive_features(patch_feats)
        B, N, D = z.shape
        return supcon_loss_fn(z.reshape(B * N, D), patch_labels.reshape(B * N))
    else:
        raise ValueError(f'Unknown aux_mode: {aux_mode!r}')

def compute_emg_features(xp, patch_size, in_ch,
                         temporal_scales=(1, 2, 4), spectral_scale=4):


    B, N, CP = xp.shape
    C, P = in_ch, patch_size
    assert CP == C * P, f'Mismatch: C*P={C*P} but CP={CP}'
    xp_3d = xp.reshape(B, N, C, P)

    def _causal_segment(s):
        if s > 1:
            pad    = torch.zeros(B, s - 1, C, P, device=xp.device, dtype=xp.dtype)
            padded = torch.cat([pad, xp_3d], dim=1)
        else:
            padded = xp_3d
        seg = padded.unfold(1, s, 1)
        return seg.permute(0, 1, 2, 4, 3).reshape(B, N, C, s * P)

    all_feats = []


    for s in temporal_scales:
        x = _causal_segment(s)
        t_rms = (x.pow(2).mean(dim=-1) + 1e-8).sqrt()
        t_mav = x.abs().mean(dim=-1)
        t_var = x.var(dim=-1)
        all_feats.append(
            torch.stack([t_rms, t_mav, t_var], dim=-1))


    x   = _causal_segment(spectral_scale)
    mag = torch.fft.rfft(x, dim=-1).abs()
    f_rms = (mag.pow(2).mean(dim=-1) + 1e-8).sqrt()
    f_mav = mag.mean(dim=-1)
    f_var = mag.var(dim=-1)
    all_feats.append(
        torch.stack([f_rms, f_mav, f_var], dim=-1))

    return torch.cat(all_feats, dim=-1).flatten(2)

def prime_factors(n):
    factors, d = [], 2
    while d * d <= n:
        while n % d == 0: factors.append(d); n //= d
        d += 1
    if n > 1: factors.append(n)
    return factors

def decimate_cascade(signal_np, labels, ds_factor):
    sig = np.asarray(signal_np, dtype=np.float64)
    for p in prime_factors(ds_factor):
        sig = scipy_decimate(sig, p, axis=1, ftype='fir', zero_phase=True)
    sig = np.ascontiguousarray(sig, dtype=np.float32)
    lbl = labels[::ds_factor][:sig.shape[1]]
    return torch.from_numpy(sig), lbl

def build_db6_paths(subj, data_root):


    paths = []
    for day in range(1, 6):
        for time in range(1, 3):
            paths.append(os.path.join(data_root, f'S{subj}_D{day}_T{time}.mat'))
    return paths

def load_session(path, ds_factor):
    data   = io.loadmat(path)
    emg    = np.array(data['emg'], dtype=np.float32).T
    stim   = data['restimulus'].reshape(-1)
    T      = min(emg.shape[1], stim.shape[0])
    emg    = emg[:, :T]; stim = stim[:T]
    mapped = np.zeros(T, dtype=np.int64)
    for orig, new in LABEL_REMAP.items():
        mapped[stim == orig] = new
    lbl = torch.from_numpy(mapped)
    if ds_factor > 1:
        sig, lbl = decimate_cascade(emg, lbl, ds_factor)
    else:
        sig = torch.from_numpy(emg)
    return sig, lbl

def find_movement_boundaries(labels):
    s = labels.view(-1)
    n = s.numel()
    if n == 0:
        empty = torch.tensor([], dtype=torch.long)
        return empty, empty, empty
    transitions = torch.where(s[1:] != s[:-1])[0] + 1
    seg_starts  = torch.cat([torch.zeros(1, dtype=torch.long), transitions])
    seg_ends    = torch.cat([transitions, torch.tensor([n], dtype=torch.long)])
    mids        = (seg_starts + seg_ends) // 2
    classes     = s[mids]
    return seg_starts, seg_ends, classes

def split_by_rep(signal, labels, train_prop):
    _, ends, classes = find_movement_boundaries(labels)
    unique_cls = torch.unique(classes)
    unique_cls = unique_cls[unique_cls != 0]
    tr_sig, tr_lbl, te_sig, te_lbl = [], [], [], []
    current = 0
    T = signal.shape[1]
    for cls in unique_cls:
        mask     = (classes == cls)
        cls_ends = ends[mask]
        n_reps   = len(cls_ends)
        if n_reps == 0:
            continue
        split_i  = min(int(np.floor(train_prop * n_reps)), n_reps - 1)
        tr_end   = (cls_ends[split_i] + 1).item()
        seg_end  = (cls_ends[-1]  + 1).item()
        tr_sig.append(signal[:, current:tr_end]);  tr_lbl.append(labels[current:tr_end])
        te_sig.append(signal[:, tr_end:seg_end]);  te_lbl.append(labels[tr_end:seg_end])
        current  = seg_end
    if current < T:
        te_sig.append(signal[:, current:T]); te_lbl.append(labels[current:T])
    if not tr_sig:
        return signal, labels, signal[:, :0], labels[:0]
    return (torch.cat(tr_sig, dim=1), torch.cat(tr_lbl),
            torch.cat(te_sig, dim=1), torch.cat(te_lbl))


def scan_common_alive(subjects, data_root, n_tr_sessions, ds_scan=10):


    common = None
    for subj in subjects:
        segs = []
        for p in build_db6_paths(subj, data_root)[:n_tr_sessions]:
            try:
                emg, _ = load_session(p, ds_scan)
                segs.append(emg)
            except Exception:
                pass
        if not segs:
            continue
        alive = torch.cat(segs, dim=1).std(dim=1) > 1e-12
        common = alive if common is None else (common & alive)
    return common


def load_subject_data(subject, data_root, ds, n_tr_sessions,
                      train_prop, post_ds_window, alive_mask=None,
                      use_filter=False):


    paths = build_db6_paths(subject, data_root)
    n_paths = len(paths)
    n_tr    = min(n_tr_sessions, n_paths)


    tr_sessions = []
    for p in paths[:n_tr]:
        try:
            tr_sessions.append(load_session(p, 1))
        except Exception:
            pass
    if not tr_sessions:
        return None


    te_sessions = []
    for p in paths[n_tr:]:
        try:
            if use_filter:
                te_sessions.append(load_session(p, ds))
            else:
                emg_full, lbl_full = load_session(p, 1)
                te_sessions.append((emg_full[:, ::ds], lbl_full[::ds]))
        except Exception:
            pass


    tr_cat = torch.cat([e for e, _ in tr_sessions], dim=1)
    if alive_mask is None:
        alive_mask = tr_cat.std(dim=1) > 1e-12
    tr_alive = tr_cat[alive_mask]
    g_mean   = tr_alive.mean(dim=1, keepdim=True)
    g_std    = tr_alive.std(dim=1,  keepdim=True).clamp_min(1e-8)
    in_ch    = int(alive_mask.sum().item())

    tr_sigs,   tr_lbls   = [], []
    te_i_sigs, te_i_lbls = [], []
    te_x_sigs, te_x_lbls = [], []
    pretrain_sigs, pretrain_lbls = [], []


    for emg, lbl in tr_sessions:
        emg_norm = (emg[alive_mask] - g_mean) / g_std
        if emg_norm.shape[1] >= post_ds_window * ds:
            pretrain_sigs.append(emg_norm); pretrain_lbls.append(lbl)
        tr_s, tr_l, te_s, te_l = split_by_rep(emg_norm, lbl, train_prop)

        if tr_s.shape[1] >= post_ds_window * ds:
            tr_sigs.append(tr_s); tr_lbls.append(tr_l)

        if te_s.shape[1] >= post_ds_window * ds:
            if use_filter:
                te_ds, te_lds = decimate_cascade(te_s, te_l, ds)
            else:
                te_ds  = te_s[:, ::ds]
                te_lds = te_l[::ds]
            if te_ds.shape[1] >= post_ds_window:
                te_i_sigs.append(te_ds); te_i_lbls.append(te_lds)


    for emg_ds, lbl_ds in te_sessions:
        emg_norm_ds = (emg_ds[alive_mask] - g_mean) / g_std
        if emg_norm_ds.shape[1] >= post_ds_window:
            te_x_sigs.append(emg_norm_ds); te_x_lbls.append(lbl_ds)

    if not tr_sigs:
        return None
    return dict(
        tr_sigs=tr_sigs,     tr_lbls=tr_lbls,
        te_i_sigs=te_i_sigs, te_i_lbls=te_i_lbls,
        te_x_sigs=te_x_sigs, te_x_lbls=te_x_lbls,
        pretrain_sigs=pretrain_sigs, pretrain_lbls=pretrain_lbls,
        in_ch=in_ch,          alive=alive_mask,
        global_mean=g_mean,   global_std=g_std,
    )


def load_subject_all_sessions(subject, data_root, alive_mask=None):


    paths = build_db6_paths(subject, data_root)
    sessions = []
    for p in paths:
        try:
            sessions.append(load_session(p, 1))
        except Exception:
            pass
    if not sessions:
        return None

    cat = torch.cat([e for e, _ in sessions], dim=1)
    if alive_mask is None:
        alive_mask = cat.std(dim=1) > 1e-12
    alive_cat = cat[alive_mask]
    g_mean = alive_cat.mean(dim=1, keepdim=True)
    g_std  = alive_cat.std(dim=1, keepdim=True).clamp_min(1e-8)
    in_ch  = int(alive_mask.sum().item())

    sigs = [(emg[alive_mask] - g_mean) / g_std for emg, _ in sessions]
    return dict(sigs=sigs, in_ch=in_ch, alive=alive_mask,
                global_mean=g_mean, global_std=g_std)


def split_70_30_id_task(sig, ds, post_ds_window, train_prop=0.7):


    T = sig.shape[1]
    split = int(T * train_prop)
    tr_sig = sig[:, :split]
    te_ds  = sig[:, split:][:, ::ds]
    tr_ok  = tr_sig.shape[1] >= post_ds_window * ds
    te_ok  = te_ds.shape[1] >= post_ds_window
    return (tr_sig if tr_ok else None), (te_ds if te_ok else None)


def sample_batch(signals_list, labels_list, batch_size,
                 window_len, patch_size, n_classes, device, ds=1):


    n_sess      = len(signals_list)
    n_patches   = window_len // patch_size
    sess_assign = torch.randint(0, n_sess, (batch_size,))
    all_windows, all_plabels = [], []
    for s in range(n_sess):
        k = int((sess_assign == s).sum().item())
        if k == 0:
            continue
        sig = signals_list[s]; lbl = labels_list[s]
        T   = sig.shape[1]

        max_start = max(0, T - (window_len - 1) * ds - 1)
        starts  = torch.randint(0, max_start + 1, (k,))

        col_idx = starts.unsqueeze(1) + torch.arange(window_len) * ds
        windows = sig[:, col_idx].permute(1, 0, 2).float()
        lbl_win = lbl[col_idx]
        lbl_pats = lbl_win[:, :n_patches * patch_size].reshape(k, n_patches, patch_size)
        lbl_oh   = F.one_hot(lbl_pats.clamp(0, n_classes - 1).long(), n_classes).float()
        plabels  = lbl_oh.sum(dim=2).argmax(dim=-1)
        all_windows.append(windows); all_plabels.append(plabels)
    return (torch.cat(all_windows, dim=0).to(device),
            torch.cat(all_plabels, dim=0).to(device))


def build_gesture_index(signals_list, labels_list, window_len, patch_size, ds=1, vote_start=0):


    valid = []
    win_full   = window_len * ds
    patch_full = patch_size * ds

    for si, (sig, lbl) in enumerate(zip(signals_list, labels_list)):
        T = sig.shape[1]
        try:
            starts_t, ends_t, classes_t = find_movement_boundaries(lbl)
        except Exception:
            continue
        for s, e, cls in zip(starts_t.tolist(), ends_t.tolist(), classes_t.tolist()):
            if cls == 0:
                continue


            for k in range(vote_start + 1):
                ws = s - k * patch_full
                if 0 <= ws and ws + win_full <= T:
                    valid.append((si, ws, int(cls)))
    return valid


def finetune_sample_batch(signals_list, labels_list, gesture_index,
                          batch_size, window_len, patch_size, n_classes, device,
                          ds=1, gesture_prob=0.75):


    n_patches  = window_len // patch_size
    n_gesture  = int(round(batch_size * gesture_prob)) if gesture_index else 0
    n_random   = batch_size - n_gesture

    results_w, results_l = [], []


    if n_gesture > 0 and gesture_index:
        idxs = torch.randint(len(gesture_index), (n_gesture,))
        for idx_val in idxs.tolist():
            si, start, _ = gesture_index[idx_val]
            sig = signals_list[si]
            lbl = labels_list[si]
            if ds > 1:
                col = start + torch.arange(window_len) * ds
            else:
                col = torch.arange(window_len) + start
            window  = sig[:, col].float()
            lbl_win = lbl[col]
            lbl_pat = lbl_win[: n_patches * patch_size].reshape(n_patches, patch_size)
            lbl_oh  = F.one_hot(lbl_pat.clamp(0, n_classes - 1).long(), n_classes).float()
            plabels = lbl_oh.sum(dim=1).argmax(dim=-1)
            results_w.append(window)
            results_l.append(plabels)


    if n_random > 0:
        rw, rl = sample_batch(signals_list, labels_list, n_random,
                              window_len, patch_size, n_classes, device, ds=ds)
        for i in range(n_random):
            results_w.append(rw[i].cpu())
            results_l.append(rl[i].cpu())

    windows = torch.stack(results_w, dim=0).to(device)
    plabels = torch.stack(results_l, dim=0).to(device)
    return windows, plabels


def evaluate_sliding_window(encoder, tcn, signals_list, labels_list,
                             window_len, patch_size, device,
                             vote_start=0, include_rest=True):
    n_patches = window_len // patch_size
    n_valid   = n_patches - vote_start
    stride    = n_valid * patch_size
    was_tr_e  = encoder.training
    was_tr_t  = tcn.training
    encoder.eval(); tcn.eval()
    correct = total = 0
    with torch.no_grad():
        for sig, lbl in zip(signals_list, labels_list):
            T = sig.shape[1]
            if T < window_len:
                continue
            starts_t, ends_t, classes_t = find_movement_boundaries(lbl)
            if classes_t.numel() == 0:
                continue
            n_full_patches = T // patch_size
            pred_by_patch  = torch.full((n_full_patches,), -1, dtype=torch.long)
            for ws in range(0, T - window_len + 1, stride):
                w      = sig[:, ws:ws + window_len].unsqueeze(0).to(device)
                feats  = encoder.encode_patches(w)
                logits = tcn(feats)
                preds  = logits[0, vote_start:].argmax(dim=-1).cpu()
                for i, pred in enumerate(preds.tolist()):
                    pi = (ws // patch_size) + vote_start + i
                    if pi < n_full_patches:
                        pred_by_patch[pi] = pred
            for s, e, cls in zip(starts_t.tolist(), ends_t.tolist(), classes_t.tolist()):
                if cls == 0 and not include_rest:
                    continue
                p0  = s // patch_size
                p1  = min(e // patch_size, n_full_patches)
                seg = pred_by_patch[p0:p1]
                seg = seg[seg >= 0]
                if seg.numel() == 0:
                    continue
                mv  = seg.mode().values.item()
                correct += int(mv == cls)
                total   += 1
    if was_tr_e: encoder.train()
    if was_tr_t: tcn.train()
    return correct / total if total > 0 else 0.0


def evaluate_patchwise(encoder, tcn, signals_list, labels_list,
                       window_len, patch_size, device, vote_start=0):


    n_patches = window_len // patch_size
    n_valid   = n_patches - vote_start
    stride    = n_valid * patch_size
    was_tr_e  = encoder.training
    was_tr_t  = tcn.training
    encoder.eval(); tcn.eval()
    correct = total = 0
    with torch.no_grad():
        for sig, lbl in zip(signals_list, labels_list):
            T = sig.shape[1]
            if T < window_len:
                continue
            for ws in range(0, T - window_len + 1, stride):
                w      = sig[:, ws:ws + window_len].unsqueeze(0).to(device)
                feats  = encoder.encode_patches(w)
                logits = tcn(feats)
                preds  = logits[0, vote_start:].argmax(dim=-1).cpu()
                for i, pred in enumerate(preds.tolist()):
                    p_start = ws + (vote_start + i) * patch_size
                    p_end   = p_start + patch_size
                    if p_end > T:
                        break
                    true_cls = lbl[p_start:p_end].mode().values.item()
                    correct += int(pred == true_cls)
                    total   += 1
    if was_tr_e: encoder.train()
    if was_tr_t: tcn.train()
    return correct / total if total > 0 else 0.0
