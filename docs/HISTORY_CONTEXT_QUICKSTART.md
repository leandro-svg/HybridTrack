# History Context Quick Start Guide

## 1. Generate History Context JSON

```bash
cd /FARM/ldibella/hybridtrack_original/docs
sbatch run_setup_history_context.sh
```

**Outputs**:
- `src/data/ann/train/history_context.json`
- `src/data/ann/validation/history_context.json`

**What it does**: Extracts (K=32, H=5, 7) agent histories from KITTI GT labels per ego frame.

---

## 2. Training Workflow

### Option A: Detection-Only (Baseline)
**Config**: `src/configs/training.yaml`
```yaml
CONTEXT:
  USE_CONTEXT: true   # Detection context enabled
  USE_HISTORY: false  # History context disabled
```

**Run**:
```bash
cd /FARM/ldibella/hybridtrack_original/src
python training_script.py
```

**Result**: Model learns with detection context only; history encoder unused.

---

### Option B: With History Context
**Config**: `src/configs/training.yaml`
```yaml
CONTEXT:
  USE_CONTEXT: true   # Detection context enabled
  USE_HISTORY: true   # History context enabled ← CHANGE THIS
```

**Run**:
```bash
cd /FARM/ldibella/hybridtrack_original/src
python training_script.py
```

**Result**: Model learns with both detection and history contexts fused via cross-attention.

---

## 3. Tracking Workflow

### Detection-Only Tracking
**Config**: `src/configs/tracking.yaml`
```yaml
CONTEXT:
  USE_CONTEXT: true   # Runtime detection context
  USE_HISTORY: false  # No history
```

**Run**:
```bash
cd /FARM/ldibella/hybridtrack_original/src
python tracking_main.py --cfg_file configs/tracking.yaml
```

---

### With History Context
**Config**: `src/configs/tracking.yaml`
```yaml
CONTEXT:
  USE_CONTEXT: true   # Runtime detection context
  USE_HISTORY: true   # Runtime history from active trajectories ← CHANGE THIS
```

**Run**:
```bash
cd /FARM/ldibella/hybridtrack_original/src
python tracking_main.py --cfg_file configs/tracking.yaml
```

---

## 4. Data Flow Summary

### Training
```
KITTI GT labels
  ↓ setup_history_context.py
history_context.json
  ↓ KITTIDataset.load_history_context()
Dataset.__getitem__ → (gt, det, history_dict)
  ↓ SystemModel.GenerateBatch()
SystemModel.HistoryContext
  ↓ DataGen()
dataset.pt (index 7: train_history)
  ↓ training_script.py
train_history → NNTrain(train_history=...)
  ↓ training.py (per time-step)
Build hist_ctx_batch (B,K,H,7), hist_mask_batch (B,K)
  ↓ LEARNABLEKF.forward()
hist_context, hist_mask → step_prior()
  ↓ LearnableKF.step_prior()
HistoryContextEncoder → CrossAttentionFusion → prior
```

### Tracking (Runtime)
```
Active trajectories (ObjectPath list)
  ↓ _prepare_history_context()
Collect last H states per trajectory
  ↓ Top-K selection by distance
hist_ctx (B,K,H,7), hist_mask (B,K)
  ↓ _lkf_prediction()
step_prior(hist_context=..., hist_mask=...)
  ↓ LearnableKF.step_prior()
HistoryContextEncoder → CrossAttentionFusion → prior
```

---

## 5. Key Configuration Parameters

**`training.yaml` / `tracking.yaml`**:
```yaml
CONTEXT:
  USE_HISTORY: false               # Enable history (false by default)
  HISTORY_LEN: 5                   # Past frames per agent
  MAX_HISTORY_AGENTS: 32           # Top-K agents
  HISTORY_DISTANCE_THRESHOLD: 50.0 # Max distance (meters)
  HISTORY_NHEAD: 4                 # Attention heads
  HISTORY_EMBED_DIM: 64            # Embedding dimension
```

---

## 6. Verification

### Check History Context File Exists
```bash
ls -lh /FARM/ldibella/hybridtrack_original/src/data/ann/train/history_context.json
ls -lh /FARM/ldibella/hybridtrack_original/src/data/ann/validation/history_context.json
```

### Check Dataset Loads History
```bash
cd /FARM/ldibella/hybridtrack_original/src
python -c "
from dataset.training_dataset import KITTIDataset
from configs.config_utils import get_cfg
cfg = get_cfg()
cfg.merge_from_file('configs/training.yaml')
ds = KITTIDataset(cfg, 'train')
print(f'Total sequences: {len(ds)}')
print(f'History sequences: {len(ds.kitti_sequences_history)}')
if len(ds.kitti_sequences_history) > 0:
    sample = ds[0]
    print(f'Sample returns {len(sample)} items (gt, det, history)')
"
```

### Check Model Accepts History
```bash
cd /FARM/ldibella/hybridtrack_original/src
python -c "
from model.LearnableKF import LEARNABLEKF, LKF
from model.model_parameters import *
from tools.batch_generation import SystemModel
from configs.config_utils import get_cfg
import torch

cfg = get_cfg()
cfg.merge_from_file('configs/training.yaml')
sys_model = SystemModel(f, Q_structure, hRotate, R_structure, 1, 1, m, n)
sys_model.InitSequence(m1x_0, m2x_0)
lkf = LEARNABLEKF(sys_model, cfg)

# Test forward with history
B, K, H = 2, 32, 5
hist_ctx = torch.randn(B, K, H, 7)
hist_mask = torch.zeros(B, K, dtype=torch.bool)
y = torch.randn(B, 7, 1)

out, prior, _ = lkf(y, hist_context=hist_ctx, hist_mask=hist_mask)
print(f'Model forward successful with history context')
print(f'Output shape: {out.shape}, Prior shape: {prior.shape}')
"
```

---

## 7. Recommended Training Schedule

1. **Baseline (Week 1)**: Train with `USE_HISTORY=false`, `USE_CONTEXT=true`
2. **Validate**: Check MOTA/HOTA metrics on validation set
3. **History Fine-tune (Week 2)**: Set `USE_HISTORY=true`, resume from best baseline checkpoint
4. **Compare**: Measure performance delta (expect +0.5–2% MOTA in dense scenes)
5. **Production**: Deploy best model; toggle history on/off via config as needed

---

## 8. Troubleshooting

### "History context file not found"
→ Run `setup_history_context.py` first

### "Index out of range" when loading dataset
→ Ensure `history_context.json` has same trajectory keys as `trajectories_ann.json`

### "History encoder not initialized"
→ Check `CONTEXT.USE_HISTORY` is `true` in config

### High memory usage
→ Reduce `MAX_HISTORY_AGENTS` or `HISTORY_LEN` in config

### No performance gain
→ History most useful in dense traffic; try evaluation on sequences 0001, 0006, 0018

---

## 9. Files Reference

| File | Purpose |
|------|---------|
| `docs/data_utils/setup_history_context.py` | Generate history JSON |
| `docs/run_setup_history_context.sh` | SLURM script for generation |
| `src/data/ann/{split}/history_context.json` | History data output |
| `src/dataset/training_dataset.py` | Load history in dataset |
| `src/tools/batch_generation.py` | Capture history in batches |
| `src/tools/training.py` | Build history tensors per step |
| `src/dataset/utils.py` | Save history in dataset.pt |
| `src/training_script.py` | Load and route history |
| `src/tracker/hybridtrack.py` | Runtime history building |
| `src/model/LearnableKF.py` | History encoder + fusion |
| `src/model/context_modules.py` | HistoryContextEncoder |
| `src/configs/training.yaml` | Training config |
| `src/configs/tracking.yaml` | Tracking config |
| `docs/HISTORY_CONTEXT_README.md` | Full documentation |

---

## 10. Contact & Support

For questions or issues:
1. Check `docs/HISTORY_CONTEXT_README.md` for detailed architecture
2. Verify config files have correct `USE_HISTORY` setting
3. Confirm history JSON files exist and are not empty
4. Test with small batch size first (N_B=4) to debug faster
