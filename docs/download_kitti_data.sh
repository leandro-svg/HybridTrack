#!/usr/bin/env bash
#SBATCH --job-name=hybridtrack_data
#SBATCH --partition=COOP
#SBATCH --output=hybridtrack_data.%j.out
#SBATCH --error=hybridtrack_data.%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4090:0
#SBATCH --mem=32G
set -euxo pipefail

# Root directory where KITTI will be stored
# You can override by calling: ./download_kitti_tracking.sh /path/to/KITTI
KITTI_ROOT="${1:-$HOME/data/KITTI}"

# Base URL for KITTI tracking (adapt to actual URLs if needed)
BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti"

# Create directory structure
mkdir -p "${KITTI_ROOT}/tracking"
cd "${KITTI_ROOT}/tracking"

echo "Downloading KITTI tracking data into: ${KITTI_ROOT}/tracking"

# -------- Training --------
mkdir -p training
cd training

# Images
if [ ! -d "image_02" ]; then
    wget -c "${BASE_URL}/data_tracking_image_2.zip" -O data_tracking_image_2.zip
    unzip -o data_tracking_image_2.zip
    rm -f data_tracking_image_2.zip
fi

# Labels
if [ ! -d "label_02" ]; then
    wget -c "${BASE_URL}/data_tracking_label_2.zip" -O data_tracking_label_2.zip
    unzip -o data_tracking_label_2.zip
    rm -f data_tracking_label_2.zip
fi

# Velodyne
if [ ! -d "velodyne" ]; then
    wget -c "${BASE_URL}/data_tracking_velodyne.zip" -O data_tracking_velodyne.zip
    unzip -o data_tracking_velodyne.zip
    rm -f data_tracking_velodyne.zip
fi

# Calibration
if [ ! -d "calib" ]; then
    wget -c "${BASE_URL}/data_tracking_calib.zip" -O data_tracking_calib.zip
    unzip -o data_tracking_calib.zip
    rm -f data_tracking_calib.zip
fi

# OXTS (IMU/GPS)
if [ ! -d "oxts" ]; then
    wget -c "${BASE_URL}/data_tracking_oxts.zip" -O data_tracking_oxts.zip
    unzip -o data_tracking_oxts.zip
    rm -f data_tracking_oxts.zip
fi

cd ..

# -------- Testing --------
mkdir -p testing
cd testing

# Images
if [ ! -d "image_02" ]; then
    wget -c "${BASE_URL}/data_tracking_image_2.zip" -O data_tracking_image_2.zip
    unzip -o data_tracking_image_2.zip
    # testing images are in the same archive; directory names differ, adjust if needed
    rm -f data_tracking_image_2.zip
fi

# Velodyne
if [ ! -d "velodyne" ]; then
    wget -c "${BASE_URL}/data_tracking_velodyne.zip" -O data_tracking_velodyne.zip
    unzip -o data_tracking_velodyne.zip
    rm -f data_tracking_velodyne.zip
fi

# Calibration
if [ ! -d "calib" ]; then
    wget -c "${BASE_URL}/data_tracking_calib.zip" -O data_tracking_calib.zip
    unzip -o data_tracking_calib.zip
    rm -f data_tracking_calib.zip
fi

# OXTS
if [ ! -d "oxts" ]; then
    wget -c "${BASE_URL}/data_tracking_oxts.zip" -O data_tracking_oxts.zip
    unzip -o data_tracking_oxts.zip
    rm -f data_tracking_oxts.zip
fi

echo "KITTI tracking download completed at: ${KITTI_ROOT}/tracking"