# Crime Detection Pipeline

End-to-end skeleton-based crime classification:
**Normal → Suspicious → Crime**

---

## Project Structure

```
crime_detection/
├── data/
│   ├── coco_to_ntu.py        # COCO→NTU mapper + normalizer + EMA smoother
│   └── dataset.py            # NPZ dataset loader + WeightedRandomSampler
├── models/
│   └── model.py              # CTR-GCN + Transformer + Weapon + FusionMLP
├── training/
│   └── train.py              # All 6 training phases
├── utils/
│   └── ucf_extractor.py      # UCF-Crime video → NPZ extractor
├── inference/
│   └── inference.py          # Real-time video inference
└── requirements.txt
```

---

## Quick Start

```bash
pip install -r requirements.txt
```

---

## Step 1 — Extract UCF-Crime skeletons

```bash
python utils/ucf_extractor.py \
    --ucf_root /data/UCF-Crime/videos \
    --output   data/crime_skeletons.npz \
    --fps 10 \
    --clip_len 64 \
    --stride 32
```

If you already have a processed NPZ, skip this step.

---

## Step 2 — Phase 2: Skeleton Pretraining

```bash
python training/train.py \
    --phase skeleton \
    --npz_path data/crime_skeletons.npz \
    --epochs 50 \
    --lr 0.01 \
    --batch_size 32 \
    --save_dir checkpoints/
```

Optional — load NTU-60 pretrained weights first:
```bash
# Download from: https://github.com/Uason-Chen/CTR-GCN (Google Drive link in README)
python training/train.py \
    --phase skeleton \
    --ntu_pretrained /path/to/ntu60_ctrgcn.pth \
    ...
```

---

## Step 3 — Phase 3: Temporal Transformer

```bash
python training/train.py \
    --phase temporal \
    --npz_path data/crime_skeletons.npz \
    --epochs 30 \
    --lr 5e-4 \
    --checkpoint checkpoints/phase2_skeleton_best.pth \
    --save_dir checkpoints/
```

---

## Step 4 — Phase 5: Fusion End-to-End

```bash
python training/train.py \
    --phase fusion \
    --npz_path data/crime_skeletons.npz \
    --epochs 30 \
    --lr 1e-4 \
    --checkpoint checkpoints/phase3_temporal_best.pth \
    --save_dir checkpoints/
```

---

## Step 5 — Phase 6: Evaluation + Export

```bash
python training/train.py \
    --phase eval \
    --npz_path data/crime_skeletons.npz \
    --checkpoint checkpoints/phase5_fusion_best.pth \
    --save_dir checkpoints/ \
    --export_onnx
```

This outputs:
- `checkpoints/thresholds.npy`     — tuned per-class thresholds
- `checkpoints/crime_detector.onnx` — exported model

---

## Step 6 — Run Inference on Video

```bash
python inference/inference.py \
    --model_path checkpoints/phase5_fusion_best.pth \
    --video_path /data/test.mp4 \
    --threshold_path checkpoints/thresholds.npy \
    --output_video output_annotated.mp4
```

---

## Run All Phases in One Command

```bash
python training/train.py \
    --phase all \
    --npz_path data/crime_skeletons.npz \
    --epochs 50 \
    --save_dir checkpoints/
```

---

## Input Format

The NPZ file must contain:

| Key | Shape | Description |
|---|---|---|
| `x_data` | `(N, T, 17, 3)` | COCO keypoints per clip |
| `y_label` | `(N,)` | 0=Normal, 1=Suspicious, 2=Crime |
| `weapon_features` | `(N, T, 6)` | Optional weapon features per frame |

If your NPZ already has 25 NTU joints: `(N, T, 25, 3)`, omit `--use_coco`.

---

## Key Technical Notes

### CTR-GCN input shape
The model expects `(N, C, T, V, M)` — channels first.
Your NPZ is `(N, T, V, C)`. The `process_clip()` function handles this transpose automatically.

### Class balance
Training uses `WeightedRandomSampler` + Focal loss to handle UCF-Crime's heavy Normal bias.

### Threshold priority (inference)
```
if p(crime) >= t_crime:    → Crime
elif p(suspicious) >= t_susp: → Suspicious
else:                         → Normal
```
This maximizes Crime recall (missing crime > false alarming suspicious).

### NTU-60 Pretraining (recommended)
Download pretrained CTR-GCN from the official repo and pass via `--ntu_pretrained`.
This gives you motion-rich features before fine-tuning on crime data.
