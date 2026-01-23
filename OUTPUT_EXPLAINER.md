## Wildfire_Impact — Outputs & Inputs Explainer (Multi-Hazard Edition)

This file explains what the training run prints and saves, where to find the generated artifacts, and what the synthetic dataset used as input looks like. Use this as a quick reference to interpret logs, metrics and checkpoints produced by `run_experiment.py`. The framework now supports both **wildfire** and **drought** event detection with dynamic class label switching.

---

## 1) Where outputs are written

- When you run training with a config, the script prints a run header like:

  Model path: `results/<model_name>/<timestamp>`

- All artifacts for that run are written under that `results/<model_name>/<timestamp>` directory. Typical contents:
  - `checkpoints/` — saved model checkpoints (may include periodic and best checkpoints).
  - `logs/` or training logs — stdout logs captured by the run (if enabled).
  - `config_used.json` (or a copy of the config) — the actual config used for the run (if the run writes it).

Tip: list the folder right after a run to find exact filenames:

```bash
ls -la results/<model_name>/<timestamp>
ls -la results/<model_name>/<timestamp>/checkpoints
```

If you do not see a checkpoint named `best_segmentation.pt` or similar, inspect the `checkpoints` folder to find the most recent `.pt` files and their timestamps.

---

## 2) What the printed log lines mean (quick)

- `===== REP X =====` — the run repeats the whole train/validation cycle `rep_times` times (config key).
- `(i) Train Loss: <value>` — training loss printed for iteration/epoch `i`. The training loop prints per-epoch (or per-iteration depending on model) aggregated train loss.
- `F1-score: <value>` — training F1 computed by the metrics module after that training epoch. This is computed from predictions and labels in the training/accumulation window.
- `Val Loss: <value>` — validation (hold-out split) loss computed during validation runs.
- `VAL F1-score: <value>` — F1 computed on validation set after a validation pass. This is the primary metric in the current scripts for measuring segmentation/change-detection quality.

Notes:
- The project uses `torchmetrics` confusion-matrix based F1 — that F1 is reported per validation pass and is the typical way to track model progress.
- Training prints progress bars for batches and shows per-epoch aggregation by default.

---

## 3) Detailed interpretation and tips

- If `VAL F1-score` is increasing over epochs, training is improving on the validation split. If `Train Loss` is decreasing but `VAL F1` drops, watch for overfitting.
- Because runs print per-epoch (and sometimes per-batch) metrics, use the printed values for quick checks. For rigorous evaluation, load the checkpoint and run `evaluate_pretrained.py` (or a dedicated evaluation script) to compute full metrics and confusion matrices.
- Checkpoint selection: look for the checkpoint file with the highest validation F1 (the run may save the "best" checkpoint). If the script saved a checkpoint named like `best_segmentation.pt` or `best_cd.pt`, use that. Otherwise pick the latest checkpoint from `checkpoints/`.

---

## 4) Where the inputs come from (dataset used in the run)

- For the local test runs we used the synthetic dataset located at:

  `data/processed/sen2_20_mod_500/`

- The split files are pickles in that directory with names referenced in `configs/config_test.json`, e.g.:

  - `synthetic_train.pkl`
  - `synthetic_val.pkl`
  - `synthetic_test.pkl`

- Each split pickle is a dictionary mapping sample ids to metadata describing the patch files. Example keys inside each sample entry:

  - `S2_before` — path to the before-image `.npy` (e.g. `.../synthetic_event_train_000_patch_000.S2_before.npy`)
  - `S2_after`  — path to the after-image `.npy`
  - `label`     — path to ground-truth label `.npy` (binary mask for changed pixels)
  - `clc_mask`  — path to a mask `.npy` included for validation and metrics (class/landcover mask)

- Array shapes (typical for the synthetic dataset used here):
  - `S2_before` and `S2_after`: numpy arrays shaped (C, H, W) where C is the number of selected bands (e.g., 20 if `sen2_20`), H and W are patch height and width (e.g. 128x128 or as in the synthetic generator).
  - `label` and `clc_mask`: arrays shaped (H, W) (binary or indexed integers depending on the dataset creation).

Config hints (important keys that affect data loading):

- `paths.dataset` — root dataset folder (we set this to `data/processed/` so the loader composes the path with `dataset_type`).
- `dataset_type` — subfolder under `paths.dataset` where the split pickles and `.npy` files live (here `sen2_20_mod_500`).
- `datasets.train/val/test` — filenames of the pickle splits (e.g. `synthetic_train.pkl`).
- `datasets.selected_bands` — which bands are loaded from the S2 arrays.
- `sen2_mean`, `sen2_std`, `mod_mean`, `mod_std` — normalization statistics expected by the dataset loader; these were added to `configs/config_test.json` for the synthetic run.

---

## 5) Reproduce the quick test run (commands)

Activate your venv and run the test config (example):

```bash
# activate venv (example for this workspace)
source venv/bin/activate

# run training using the test config
python run_experiment.py --config configs/config_test.json
```

If you prefer a quicker test, edit `configs/config_test.json` and set e.g. `train.epochs` to `2` and `rep_times` to `1` to see a short run.

---

## 6) How to evaluate a saved checkpoint

1. Find the checkpoint file under `results/<model_name>/<timestamp>/checkpoints/`.
2. Run the evaluation script that loads the checkpoint (the repository includes `evaluate_pretrained.py`):

```bash
source venv/bin/activate
python evaluate_pretrained.py --checkpoint results/<model_name>/<timestamp>/checkpoints/<checkpoint_file>.pt \
    --config configs/config_test.json --split test
```

Adjust CLI arguments according to how `evaluate_pretrained.py` expects them (it may need a `--config` and `--split` argument; check its `--help`).

---

## 7) Common warnings you may see (and what they mean)

- `pin_memory` MPS warning: on macOS Apple silicon the DataLoader `pin_memory` flag is not supported by MPS; it will be ignored. This is harmless for CPU/MPS runs.
- `torch.cuda.amp.autocast` deprecation: the code uses the older signature. It's a deprecation warning; it doesn't break training on CPU.
- `WindowsPath` in pickles: pickles created on Windows can contain `WindowsPath` objects that break on macOS; re-create the split pickles on your machine to avoid this.

---

## 8) Quick troubleshooting checklist

- If train fails at dataset load: confirm `paths.dataset`, `dataset_type` and `datasets.*.pkl` resolve to actual files.
- If metrics throw an error about shapes: inspect `label` array shapes and make sure they match model output shape.
- If evaluation fails to find checkpoint: list `results/<model_name>/<timestamp>/checkpoints` to find available `.pt` files.

---

## 9) Next steps / suggestions

- Add a tiny script to plot `VAL F1` over epochs (the run directory can be parsed to build a small CSV from logged values).
- Add an explicit `results/<model>/<timestamp>/README.md` writer in `run_experiment.py` so each run captures the config used and the checkpoint filenames saved.

If you want, I can also:
- generate a small Jupyter notebook that loads a chosen checkpoint and visualizes predictions vs labels for a handful of test patches, or
- add a helper script to extract the best checkpoint automatically and run `evaluate_pretrained.py`.

---

File created on: 2026-01-15
