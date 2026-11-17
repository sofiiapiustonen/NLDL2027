import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

RUNS = {
    'jepa_frozen':    ['jepa_pretrain_and_classify.py', '2'],
    'jepa_finetuned': ['jepa_pretrain_and_classify.py', '3'],
    'mae_frozen':     ['mae_pretrain_and_classify.py', '2'],
    'mae_finetuned':  ['mae_pretrain_and_classify.py', '3'],
    'encoder_tcn':    ['baseline_encoder_tcn.py'],
    'tcn_only':       ['baseline_tcn_only.py'],
}

SHARE_PRETRAIN = {
    'jepa_finetuned': ('jepa_cross', 'setting_2_jepa_only_frozen', 'setting_3_jepa_only_unfrozen'),
    'mae_finetuned':  ('mae_cross', 'setting_2_mae_only_frozen', 'setting_3_mae_only_unfrozen'),
}


def share_pretrain(name, seed):


    if name not in SHARE_PRETRAIN:
        return
    folder, src_setting, dst_setting = SHARE_PRETRAIN[name]
    base = os.path.join(HERE, 'results', folder, f'seed_{seed}')
    src = os.path.join(base, src_setting, 'pretrained_final.pt')
    dst = os.path.join(base, dst_setting, 'pretrained_final.pt')
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f'  reusing pre-trained encoder: {src}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', default=os.environ.get('DATA_ROOT', os.path.expanduser('~/Desktop/ninapro')),
                    help='folder with the NinaPro DB6 files S{subj}_D{day}_T{time}.mat')
    ap.add_argument('--seeds', type=int, nargs='+', default=[41, 42, 43])
    ap.add_argument('--only', nargs='+', choices=list(RUNS), default=list(RUNS),
                    help='run only these (default: all)')
    args = ap.parse_args()

    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    if not os.path.isdir(data_root):
        sys.exit(f'data root not found: {data_root}  (pass --data-root)')

    log_dir = os.path.join(HERE, 'results', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    failed = []
    for seed in args.seeds:
        for name in args.only:
            share_pretrain(name, seed)
            cmd = [sys.executable, '-u'] + RUNS[name]
            env = dict(os.environ, DATA_ROOT=data_root, SEED=str(seed))
            log = os.path.join(log_dir, f'{name}_seed{seed}.txt')
            print(f'\n{"=" * 72}\n>>> {name}  seed {seed}   ({" ".join(RUNS[name])})\n    log: {log}\n{"=" * 72}',
                  flush=True)
            t0 = time.time()
            with open(log, 'a') as fh:
                fh.write(f'\n=== {time.ctime()}  {name}  seed {seed} ===\n')
                fh.flush()
                proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    sys.stdout.write(line)
                    fh.write(line)
                proc.wait()
            dt = (time.time() - t0) / 60
            if proc.returncode != 0:
                failed.append(f'{name} seed {seed}')
                print(f'!!! {name} seed {seed} FAILED (exit {proc.returncode}) after {dt:.1f} min', flush=True)
            else:
                print(f'--- {name} seed {seed} done in {dt:.1f} min', flush=True)

    print('\nAll runs finished.' if not failed else '\nFailed runs:\n  ' + '\n  '.join(failed))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
