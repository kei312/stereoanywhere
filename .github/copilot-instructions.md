### Goal
Help contributors and AI coding agents make small, safe, and high‑impact changes to the StereoAnywhere codebase (CVPR2025). Focus on runnable edits, debugging training/evaluation scripts, and model wiring.

### High level architecture (short)
- Dual-branch model: a stereo branch (cost-volume + RAFT-like recurrent update) and a monocular prior branch (Depth‑Anything‑V2). Entry point: `models/stereoanywhere/stereoanywhere.py`.
- Monocular model loader: `models/depth_anything_v2/__init__.py` exposes `get_depth_anything_v2(checkpoint_path, encoder, map_location)` used by `train.py`, `test.py`, and `mono_sceneflow.py`.
- Data handling: `dataloaders/fetch_dataloader(args)` (see `dataloaders/__init__.py`) selects dataset classes under `dataloaders/` (e.g. `MiddleburyDataset`, `FlyingThingsDataset`, `MonoTrapDataset`).

### Typical developer workflows (commands & examples)
- Install dependencies: `pip install -r requirements.txt`. Demo extras: `demo/requirements_demo.txt`.
- Precompute monocular depth for Sceneflow before training: `python mono_sceneflow.py --datapath <SCENEFLOW_PATH> --monomodel DAv2 --loadmonomodel <MONO_MODEL_PATH>` (see `README.md` and `mono_sceneflow.py`).
- Quick local training (example): edit `run_train.sh` paths then run it. Equivalent direct command (minimal):
  - `python train.py --dataset sceneflow --datapath <FLY;MONK;DRIV> --loadmonomodel <MONO_MODEL> --loadmodel <RAFT_INIT> --model stereoanywhere --batch_size 2 --epochs 3`
- Run evaluation with pretrained weights (example shown in README):
  - `python test.py --datapath <DATAPATH> --dataset <middlebury|booster|kitti2015|...> --stereomodel stereoanywhere --loadstereomodel <STEREO_MODEL_PATH> --monomodel DAv2 --loadmonomodel <MONO_MODEL_PATH> --iscale 1 --oscale 1 --iters 32 --vol_n_masks 8 --use_aggregate_mono_vol`

### Project-specific conventions and gotchas
- Many scripts assume batch-size 1 at test time and that the monocular model is used to generate a per-image depth prior scaled to [0,1]. See `test.py` inference block and `_input_size_*` heuristics.
- Preloading monocular predictions: `dataloaders.fetch_dataloader` supports `args.preload_mono` to pass `mono` filepaths into dataset classes (used for sceneflow training/testing). If you add a dataset, expose a `mono` argument and respect preloaded files.
- Model freezing: training supports `--freeze_for_finetuning` and `--things_to_freeze fnet|cnet|monoagg` which calls `StereoAnywhere.freeze_for_finetuning()` (see `models/stereoanywhere/stereoanywhere.py`). Keep layer names matching this dict when refactoring.
- Mixed precision and device flags: scripts use `--mixed_precision` and `--no-cuda` with torch.autocast/GradScaler. When changing numeric code, test both AMP on/off to catch precision bugs.
- Input padding: `test.py` pads images to multiples of 32 (inside `run`) — be careful when changing spatial ops or adding layers that alter expected downsampling factors.

### Key files to open for targeted edits
- Model and blocks: `models/stereoanywhere/stereoanywhere.py`, `models/stereoanywhere/extractor.py`, `models/stereoanywhere/update.py`, `models/stereoanywhere/corr.py`, `models/stereoanywhere/hourglass.py`
- Monocular adapter: `models/depth_anything_v2/__init__.py` and `models/depth_anything_v2/dpt.py` (loads checkpoints and defines infer utilities used by scripts).
- Train / test flows: `train.py`, `test.py`, `test_monotrap.py`, `run_train.sh`, `mono_sceneflow.py`.
- Data loaders: `dataloaders/__init__.py` and dataset implementations under `dataloaders/` (e.g., `middlebury_dataset.py`, `flyingthings_dataset.py`, `monotrap_dataset.py`).
- Utilities and visualizations: `utils.py` (normal estimation, warping, sgm fallback), `losses.py` (loss composition used in training).

### What an AI agent can change safely (low-risk edits)
- Small fixes: argument parsing help text, default hyperparameters, logging messages, docstrings, minor refactors inside a single file.
- Add assertion or argument validation in CLI scripts (`train.py`/`test.py`) to make paths and device flags explicit.
- Bug fixes that preserve API: e.g., correct tensor shapes after padding/unpadding, small tensor dtype fixes (float32/float64), or replace deprecated torch APIs.

### Riskier edits — require running training/test locally
- Any change to model forward pass, volume generation, or corr implementation. These require a tiny reproducible test (use `test.py` on one small dataset/sample) and verifying output shapes / metrics.

### Minimal quick checks to run after changes
1. Lint/import sanity: open the modified file to ensure it imports from local modules (no circular imports).  
2. Run `python -c 'import torch; import models.stereoanywhere; print("ok")'` to quickly check model import.  
3. Run `python test.py --datapath <small_dataset_path> --dataset <middlebury> --loadstereomodel <CHECKPOINT> --loadmonomodel <MONO>` with `--valsize 1 --tries 1 --mixed_precision` to validate end-to-end shape and runtime.

### Examples to copy/paste when editing
- Load mono model: `mono = get_depth_anything_v2(checkpoint_path, encoder='vitl', map_location=device)` (see `models/depth_anything_v2/__init__.py`).
- Freeze layers: call `model.freeze_for_finetuning()` if script exposes `--freeze_for_finetuning` (see `train.py`).

If anything in these instructions is unclear or missing (e.g. dataset paths, expected checkpoint formats, or demo runtime steps), tell me what to expand and I will iterate.
