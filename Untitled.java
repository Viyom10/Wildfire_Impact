(venv) arnavbajaj@Arnavs-MacBook-Air-5 Wildfire_Impact % python create_synthetic_dataset.py --event_type wildfire
python run_experiment.py --config configs/config_eval_synthetic.json
============================================================
Creating Minimal Test Dataset
============================================================

Creating synthetic patches...

Creating train split (10 samples) for wildfire...
  ✓ Created synthetic_event_train_000_patch_000 (positive)
  ✓ Created synthetic_event_train_001_patch_000 (negative)
  ✓ Created synthetic_event_train_002_patch_000 (positive)
  ✓ Created synthetic_event_train_003_patch_000 (negative)
  ✓ Created synthetic_event_train_004_patch_000 (positive)
  ✓ Created synthetic_event_train_005_patch_000 (negative)
  ✓ Created synthetic_event_train_006_patch_000 (positive)
  ✓ Created synthetic_event_train_007_patch_000 (negative)
  ✓ Created synthetic_event_train_008_patch_000 (positive)
  ✓ Created synthetic_event_train_009_patch_000 (negative)
  ✓ Saved data/processed/sen2_20_mod_500/synthetic_train.pkl

Creating val split (3 samples) for wildfire...
  ✓ Created synthetic_event_val_000_patch_000 (positive)
  ✓ Created synthetic_event_val_001_patch_000 (negative)
  ✓ Created synthetic_event_val_002_patch_000 (positive)
  ✓ Saved data/processed/sen2_20_mod_500/synthetic_val.pkl

Creating test split (3 samples) for wildfire...
  ✓ Created synthetic_event_test_000_patch_000 (positive)
  ✓ Created synthetic_event_test_001_patch_000 (negative)
  ✓ Created synthetic_event_test_002_patch_000 (positive)
  ✓ Saved data/processed/sen2_20_mod_500/synthetic_test.pkl

============================================================
✅ Synthetic wildfire dataset created successfully!
============================================================

Dataset location: data/processed

Split files:
  - data/processed/sen2_20_mod_500/synthetic_train.pkl
  - data/processed/sen2_20_mod_500/synthetic_val.pkl
  - data/processed/sen2_20_mod_500/synthetic_test.pkl

Total patches: 16
Dataset size: ~36.0 MB

You can now run evaluation with:
  python run_experiment.py --config configs/config_eval_synthetic.json
============================================================
/Users/arnavbajaj/Desktop/Wildfire_Impact/venv/lib/python3.13/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
--- Testing model for models/pretrained/bam_cd_best.pt/best_segmentation.pt ---
Using event type: wildfire
Traceback (most recent call last):
  File "/Users/arnavbajaj/Desktop/Wildfire_Impact/run_experiment.py", line 192, in <module>
    checkpoint = torch.load(ckpt_path, map_location=device)
  File "/Users/arnavbajaj/Desktop/Wildfire_Impact/venv/lib/python3.13/site-packages/torch/serialization.py", line 1484, in load
    with _open_file_like(f, "rb") as opened_file:
         ~~~~~~~~~~~~~~~^^^^^^^^^
  File "/Users/arnavbajaj/Desktop/Wildfire_Impact/venv/lib/python3.13/site-packages/torch/serialization.py", line 759, in _open_file_like
    return _open_file(name_or_buffer, mode)
  File "/Users/arnavbajaj/Desktop/Wildfire_Impact/venv/lib/python3.13/site-packages/torch/serialization.py", line 740, in __init__
    super().__init__(open(name, mode))
                     ~~~~^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'models/checkpoints/0/best_segmentation.pt'