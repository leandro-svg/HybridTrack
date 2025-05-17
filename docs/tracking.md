# Tracking Guide

This document explains how to configure and launch the tracking process for HybridTrack.

## 1. Prepare Data

- Ensure your data is set up as described in the [Data Preparation Guide](create_data.md).

## 2. Adjust Tracking Configuration

- Open the configuration file:
  ```
  src/configs/tracking.yaml
  ```
- Update all relevant paths, especially:
  - `dataset_path`: Path to your KITTI tracking data
  - `detections_path`: Path to your detections folder
  - `save_path`: Where tracking results will be saved
- Example:
  ```yaml
  dataset_path: "src/data/KITTI/tracking/training"
  detections_path: "/src/data/detections/virconv/training"
  save_path: "src/evaluator/evaluation/results/sha_key/data"
  ```
- Make sure any other path fields are also correct for your environment.

## 3. Download and Place Model Weights

- Train your own model or Download the HybridTrack PyTorch model weights from [this link](https://drive.google.com/file/d/1beFjycNjTtb2nDDf0vteHp1NNbR4lrvR/view?usp=sharing).
- Place the downloaded `.pth` file in:
  ```
  src/result/hybridtrack/online/model_checkpoint
  ```
## 4. Launch Tracking

- From the project root, run:
  ```bash
  python src/run_tracking.py --cfg_file src/configs/tracking.yaml
  ```

---

**Note:**
- If you encounter errors related to missing files or directories, double-check all paths in `tracking.yaml` and your data setup.
