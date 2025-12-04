import json
import os

from tracker.obectPath import LEARNABLEKF
# Use SLURM allocated GPUs if available, otherwise default to GPU 1
if 'SLURM_JOB_GPUS' not in os.environ and 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from torch.utils.data import DataLoader
from tools.batch_generation import SystemModel
from dataset.utils import DataGen, DataGen_eval
from configs.config_utils import general_settings
from dataset.training_dataset import KITTIDataset
from tools.training import TrainingPipeline
from datetime import datetime
import numpy as np
import logging
from model.model_parameters import m1x_0, m2x_0, m, n,\
f, h, hRotate, Q_structure, R_structure
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cfg = general_settings()

SEED = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

logging.info("Pipeline Start")

today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
logging.info(f"Current Time = {strTime}")

train_bool = True  # Set as needed
load_data = True

# Check for force CPU mode via environment variable
force_cpu = os.environ.get('FORCE_CPU', 'false').lower() in ['true', '1', 'yes']

if cfg.TRAINER.USE_CUDA and not force_cpu:
   if torch.cuda.is_available():
      device = torch.device('cuda')
      logging.info(f"Using GPU - CUDA device count: {torch.cuda.device_count()}")
      logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
      logging.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
   else:
      logging.error("No GPU found, but USE_CUDA is True.")
      logging.error(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
      logging.error(f"SLURM_JOB_GPUS: {os.environ.get('SLURM_JOB_GPUS', 'Not set')}")
      logging.warning("Falling back to CPU mode. Set FORCE_CPU=true to suppress this warning.")
      device = torch.device('cpu')
else:
    device = torch.device('cpu')
    if force_cpu:
        logging.info("Using CPU (forced via FORCE_CPU environment variable)")
    else:
        logging.info("Using CPU")

DatafolderName = os.path.join(cfg.DATASET.ROOT, 'src', 'data', 'checkpoints')

# noise q and r
Q = Q_structure
R = R_structure

dataFileName = ['dataset.pt']
if train_bool:
    index_datafile = 0
else:
    index_datafile = 1
data_file_path = os.path.join(DatafolderName, dataFileName[index_datafile])

if load_data:
    if train_bool:
        train_dataset = eval(cfg.DATASET.NAME)(cfg, mode=cfg.DATASET.MODE)
        val_dataset = eval(cfg.DATASET.NAME)(cfg, mode='validation')
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=0)
    else:
        test_dataset = eval(cfg.DATASET.NAME)(cfg, mode='validation')
        test_dataloader = DataLoader(test_dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=0)

sys_model = SystemModel(f, Q, hRotate, R, cfg.DATASET.SEQ_LEN, cfg.TRAINER.T_TEST, m, n)
sys_model.InitSequence(m1x_0, m2x_0)
logging.info("Starting Data Generation...")
if load_data:
    if not os.path.exists(DatafolderName):
        os.makedirs(DatafolderName)
        logging.info(f"Created directory: {DatafolderName}")
    if train_bool:
        DataGen(cfg, sys_model, train_dataloader, val_dataloader, data_file_path)
    else:
        DataGen_eval(cfg, sys_model, test_dataloader, data_file_path)
logging.info(f"Data Generation Complete. Data saved to: {data_file_path}")

logging.info(f"Loading data from: {data_file_path}")
if train_bool:
    loaded_data = torch.load(data_file_path, map_location='cpu')
    train_input = loaded_data[0]
    train_target = loaded_data[1]
    cv_input = loaded_data[2]
    cv_target = loaded_data[3]
    # New format: [train_input, train_target, cv_input, cv_target, train_init, cv_init,
    #              train_context, train_history, cv_context, cv_history]
    train_context = loaded_data[6] if len(loaded_data) > 6 else None
    train_history = loaded_data[7] if len(loaded_data) > 7 else None
    cv_context = loaded_data[8] if len(loaded_data) > 8 else None
    cv_history = loaded_data[9] if len(loaded_data) > 9 else None
    
    logging.info(f"Number of training samples: {len(train_input)}")
    logging.info(f"Number of cross-validation samples: {len(cv_input)}")
    if train_context is not None:
        non_empty_ctx = sum(1 for c in train_context if c and len(c) > 0)
        logging.info(f"Loaded detection context for {non_empty_ctx}/{len(train_context)} training samples")
    if train_history is not None:
        non_empty_hist = sum(1 for h in train_history if h and 'history_per_frame' in h)
        logging.info(f"Loaded history context for {non_empty_hist}/{len(train_history)} training samples")
    if cv_context is not None:
        non_empty_cv_ctx = sum(1 for c in cv_context if c and len(c) > 0)
        logging.info(f"Loaded detection context for {non_empty_cv_ctx}/{len(cv_context)} CV samples")
    if cv_history is not None:
        non_empty_cv_hist = sum(1 for h in cv_history if h and 'history_per_frame' in h)
        logging.info(f"Loaded history context for {non_empty_cv_hist}/{len(cv_history)} CV samples")
    
    # Handle backward compatibility: create dummy contexts if needed
    if train_context is not None and cv_context is None:
        logging.warning("Training uses context but cv_context is None - creating empty context for CV")
        cv_context = [[torch.zeros((0, 8), dtype=torch.float32) for _ in range(cfg.DATASET.SEQ_LEN)] for _ in range(len(cv_input))]
    if train_history is not None and cv_history is None:
        logging.warning("Training uses history but cv_history is None - creating empty history for CV")
        cv_history = [{} for _ in range(len(cv_input))]
else:
    loaded_data = torch.load(data_file_path, map_location='cpu')
    test_input = loaded_data[0]
    test_target = loaded_data[1]
    logging.info(f"Number of test samples: {len(test_input)}")

sys_model_partial = SystemModel(f, Q, h, R, cfg.DATASET.SEQ_LEN, cfg.TRAINER.T_TEST, m, n)
sys_model_partial.InitSequence(m1x_0, m2x_0)

LKF_model = LEARNABLEKF(sys_model, cfg)
LKF_Pipeline = TrainingPipeline(strTime, "LKF", "LKF")
LKF_Pipeline.set_ss_model(sys_model_partial)
LKF_Pipeline.set_model(LKF_model)
LKF_Pipeline.set_training_params(cfg)

# Diagnostic logging for context usage
if train_bool:
    use_context = hasattr(LKF_model.LKF_model, 'use_context') and LKF_model.LKF_model.use_context
    logging.info(f"[Diagnostic] USE_CONTEXT={use_context}, train_context_present={train_context is not None}, cv_context_present={cv_context is not None}")
    if use_context and (train_context is None or cv_context is None):
        logging.error("WARNING: Context enabled but context data missing - performance will degrade!")
type_network = 'hybridtrack'
type_tracking = 'online'

path_results_base = os.path.join(cfg.DATASET.ROOT, 'src', 'result', type_network, type_tracking, os.path.basename(data_file_path).replace('.pt', ''))
path_results = os.path.join(path_results_base, f"{strTime}_T{cfg.DATASET.SEQ_LEN}_Ttest{cfg.TRAINER.T_TEST}_nSteps{cfg.TRAINER.EPOCH}_mBtach{cfg.TRAINER.BATCH_SIZE}_lr{cfg.TRAINER.LR}_wd{cfg.TRAINER.WD}")
os.makedirs(path_results, exist_ok=True)

path_results_code = os.path.join(path_results, 'code')
os.makedirs(path_results_code, exist_ok=True)

if train_bool:
    shutil.copy(__file__, os.path.join(path_results_code, "main.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__),"model", "LearnableKF.py"), os.path.join(path_results_code, "model.py"))
    shutil.copy(os.path.join(os.path.dirname(__file__),"tools", "training.py"), os.path.join(path_results_code, "pipeline.py"))

if train_bool:
    path_results_config = os.path.join(path_results, 'config')
    os.makedirs(path_results_config, exist_ok=True)
    import yaml
    with open(os.path.join(path_results_config, 'config.yaml'), "w") as file:
        yaml.dump(cfg, file)
    logging.info(f"Configuration saved to {os.path.join(path_results_config, 'config.yaml')}")

    [MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = LKF_Pipeline.NNTrain(sys_model_partial,
                                                                                                            cv_input,
                                                                                                            cv_target,
                                                                                                            train_input,
                                                                                                            train_target,
                                                                                                            path_results, cfg,
                                                                                                            train_context=train_context,
                                                                                                            cv_context=cv_context,
                                                                                                            train_history=train_history,
                                                                                                            cv_history=cv_history)
else:
    path_results_weight = os.path.join(path_results, 'weights')
    path_results_val = os.path.join(path_results, 'val_test')
    os.makedirs(path_results_val, exist_ok=True)

    if not os.path.exists(path_results_weight):
        os.makedirs(path_results_weight)
        logging.info(f"Created weights directory (or it already existed): {path_results_weight}")

    [MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, LKF_out] = LKF_Pipeline.NNTest(sys_model_partial,
                                                                                                test_input,
                                                                                                test_target,
                                                                                                path_results_val,
                                                                                                path_results_weight,
                                                                                                0,
                                                                                                cfg, test_init=False)

