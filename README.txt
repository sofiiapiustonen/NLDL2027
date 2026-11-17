JEPA vs MAE for EMG gesture recognition (NinaPro DB6).

How to run everything:
    pip install -r requirements.txt
    python3 run_all_experiments.py --data-root /path/to/ninapro

Files:
    run_all_experiments.py         runs all models below for seeds 41, 42, 43
    jepa_pretrain_and_classify.py  JEPA pre-training + classification (arg 2 = frozen, 3 = fine-tuned)
    mae_pretrain_and_classify.py   MAE pre-training + classification (arg 2 = frozen, 3 = fine-tuned)
    baseline_encoder_tcn.py        supervised baseline: Transformer encoder + TCN
    baseline_tcn_only.py           supervised baseline: TCN only
    models.py                      data loading, models, training and evaluation

Results are saved in results/ (accuracy per subject in results.npz, logs in results/logs/).
