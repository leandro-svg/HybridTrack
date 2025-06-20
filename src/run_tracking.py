import argparse
import sys
import os
from tracking_main import tracking_val_seq


parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--cfg_file', type=str, default="src/configs/tracking.yaml",
                    help='specify the config for tracking')
args = parser.parse_args()
tracking_val_seq(args)