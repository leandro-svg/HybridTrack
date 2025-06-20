# Training Guide

This document explains how to configure and launch the training process for HybridTrack.

## 1. Prepare Data

- Ensure your data is set up as described in the [Data Preparation Guide](create_data.md).

## 2. Adjust Training Configuration

- Open the configuration file:
  ```
  src/configs/training.yaml
  ```
- Update all relevant paths, especially `DATASET.ROOT`, to point to your data location. For example:
  ```yaml
  DATASET:
    ROOT: "src/data"
  ```
- Make sure any other path fields (such as `MODEL.SAVE_PATH`) are also correct for your environment.

## 3. Launch Training

- From the project root, run:
  ```bash
  python src/training_script.py --config src/configs/training.yaml
  ```

---

**Note:**
- If you encounter errors related to missing files or directories, double-check all paths in `training.yaml` and your data setup.
