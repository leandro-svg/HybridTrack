# Data Generation Guide

This document explains how to set up the KITTI tracking dataset and generate the required annotation files for this project.

## 1. Download KITTI Tracking Dataset

- Download the KITTI tracking dataset from the [official KITTI website](http://www.cvlibs.net/datasets/kitti/eval_tracking.php).
- Download both the `training` and `testing` sets.
- Download and unzip the pose data from the following link: [Google drive for pose data](https://drive.google.com/file/d/1o-ay2FhlOEnKFmqXMWnbH7e6gId8L7P9/view?usp=sharing).                                           
- Unzip the dataset so you have the following structure:
  ```
  KITTI/
    tracking/
      training/
        image_02/
        label_02/
        pose/
        ...
      testing/
        image_02/
        pose/
        ...
  ```

## 2. Prepare Folder Structure

Your data directory should look like this:
```
src/data/
  KITTI/                # (symlink to your KITTI folder)
    tracking/
      training/
        calib/
        image_02/
        label_02/
        pose/
        velodyne/
        oxts/
      testing/
        calib/
        image_02/
        pose/
        velodyne/
        oxts/
      detections/ (See step 4)
        virconv/
          training/
            0000/
            0001/
            ...
          testing/
  ann/                   # (will be created by the script)
```

- **Symlink the KITTI folder**:  
  If your KITTI data is elsewhere, create a symbolic link in `src/data`:
  ```bash
  ln -s /path/to/KITTI src/data/KITTI
  ```

## 3. Download Detections

- Download the detections from [this Google Drive folder](https://drive.google.com/drive/folders/1SrVjCsL1sT0x5h2m50UM_XXifpBD9HMD?usp=sharing) and place the `virconv` folder inside `src/data/KITTI/tracking/detections`.

## 4. Generate Annotations

- Run the following script to generate the annotation files:
  ```bash
  python docs/data_utils/setup_trajectory.py
  ```
- This will create the `ann` folder in `src/data/` with the required annotation files for training and validation.

---

## 5. Label for evaluation
  - Add tje tracking label_02 in 'src/evaluator/evaluation/data/tracking/label_02/'. 
**Note:**  
- Make sure all paths are correct and match the structure above.
- If you encounter any issues, check that all folders and files are in the expected locations.
