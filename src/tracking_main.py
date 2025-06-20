import math
from dataset.tracking_dataset import KittiTrackingDataset
from dataset.tracking_dataset_utils import velo_to_cam
# from castrack.tracker.tracker_2d import Tracker3D
###
##
from tracker.hybridtrack import HYBRIDTRACK
import time
import tqdm
import os
from configs.config_utils import cfg, cfg_from_yaml_file
from tracker.box_op_2d import *
import numpy as np
import argparse
import math
from evaluator.evaluation_HOTA.scripts.run_kitti import eval_kitti
import matplotlib.pyplot as plt
from tracker.post_process import filter_and_interpolate_trajectory  # Import post-processing

def track_one_seq(seq_id,config):
    """
    tracking one sequence
    Args:
        seq_id: int, the sequence id
        config: config
    Returns: dataset: KittiTrackingDataset
             tracker: Tracker3D
             all_time: float, all tracking time
             frame_num: int, num frames
    
    """
    print(f"TRACKING SEQIN {seq_id}")
    dataset_path = config.dataset_path
    detections_path = config.detections_path
    tracking_type = config.tracking_type
    detections_path += "/" + str(seq_id).zfill(4)
    before_alloc = torch.cuda.memory_allocated()
        
    tracker = HYBRIDTRACK(box_type="Kitti", tracking_features=False, config = config)
    dataset = KittiTrackingDataset(dataset_path, seq_id=seq_id, ob_path=detections_path,type=[tracking_type])

    all_time = 0
    frame_num = 0
    total_number_objets = 0
    for i in range(len(dataset)):
        P2, V2C, points, image, objects, det_scores, pose = dataset[i]
        total_number_objets += len(objects)
        mask = det_scores>config.input_score
        objects = objects[mask]
        det_scores = det_scores[mask]

        start = time.time()

        tracker.tracking(objects[:,:7],
                             features=None,
                             scores=torch.tensor(det_scores),
                             pose=pose,
                             timestamp=i, v2c=V2C, p2=P2)
        end = time.time()
        all_time+=end-start
        frame_num+=1
    after_alloc = torch.cuda.memory_allocated()
    print(f"Memory used for sequence {seq_id}:", after_alloc - before_alloc, "bytes")  
    print(f"Processing Framerate for sequence {seq_id}:",frame_num/all_time, "FPS")  
    print(f"Detected Objects per frame for sequence {seq_id}:",total_number_objets/frame_num, "Objets per Frame")  
    print(f"Total detected Objects for sequence {seq_id}:",total_number_objets, "FPS")  
    tracker.remove_model_from_gpu()
    torch.cuda.empty_cache()
    return dataset, tracker, all_time, frame_num

def smoothstep(t):
    """
    Smoothstep function that maps t in [0, 1] to a smooth value also in [0, 1].
    It accelerates slowly at the beginning and decelerates at the end.
    """
    return 3 * t**2 - 2 * t**3

def interpolate_state(prev_state, next_state, t):
    """
    Interpolates between two vehicle states using smoothstep.
    
    Parameters:
    - prev_state: array-like of vehicle parameters [x, y, z, w, h, l] at start.
    - next_state: array-like of vehicle parameters [x, y, z, w, h, l] at end.
    - t: float between 0 and 1, the interpolation factor.
    
    Returns:
    - interpolated_state: NumPy array representing the interpolated state.
    """
    prev_state = np.array(prev_state, dtype=float)
    next_state = np.array(next_state, dtype=float)
    
    # Ensure t is within bounds
    t = np.clip(t, 0.0, 1.0)
    
    # Apply smoothstep to get the non-linear interpolation factor.
    smooth_t = smoothstep(t)
    
    # Interpolate each parameter using the smooth factor.
    interpolated_state = prev_state + (next_state - prev_state) * smooth_t
    
    return interpolated_state



def save_one_seq(dataset,
                 seq_id,
                 tracker,
                 config):
    """
    saving tracking results
    Args:
        dataset: KittiTrackingDataset, Iterable dataset object
        seq_id: int, sequence id
        tracker: Tracker3D
    """

    save_path = config.save_path
    tracking_type = config.tracking_type
    s =time.time()
    tracks = tracker.post_processing(config)
    proc_time = s-time.time()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_name = os.path.join(save_path,str(seq_id).zfill(4)+'.txt')

    frame_first_dict = {}
    for ob_id in tracks.keys():
        track = tracks[ob_id]
        if config.latency == 1 and config.post_process_interpolation:
            filter_and_interpolate_trajectory(track.trajectory)
        for frame_id in track.trajectory.keys():
            ob = track.trajectory[frame_id]
            average_score = track.total_detected_score/track.total_detections
            
            if config.latency == 0 :
                if ob.score is None or ob.score<config.post_score_online:
                    continue
            elif config.latency == 1:
                if ob.timestamp > track.last_updated_timestamp:
                    continue
               
                if average_score < config.post_score_offline:
                    continue


            if frame_id in frame_first_dict.keys():
                frame_first_dict[frame_id][ob_id]=(np.array(ob.updated_state.T.detach().numpy()),ob.score)
            else:
                frame_first_dict[frame_id]={ob_id:(np.array(ob.updated_state.T.detach().numpy()),ob.score)}

    with open(save_name,'w+') as f:
        for i in range(len(dataset)):
            P2, V2C, points, image, _, _, pose = dataset[i]
            new_pose = np.asmatrix(pose).I
            if i in frame_first_dict.keys():
                objects = frame_first_dict[i]

                for ob_id in objects.keys():
                    updated_state,score = objects[ob_id]
                    updated_state, score = updated_state, score.cpu().numpy()
                    box_template = np.zeros(shape=(1,7))
                    box_template[0,0:3]=updated_state[0:3]
                    box_template[0,3:7]=updated_state[3:7]

                    box = register_bbs_numpy_initial(box_template,new_pose)
                    box = box_template
                    box[:, 6] = -box[:, 6] - np.pi / 2
                    box[:, 2] -= box[:, 5] / 2
                    box[:,0:3] = velo_to_cam(box[:,0:3],V2C)[:,0:3]

                    box = box[0]

                    box2d = bb3d_2_bb2d_numpy(box,P2)

                    print('%d %d %s -1 -1 -10 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                          % (i,ob_id,tracking_type,box2d[0][0],box2d[0][1],box2d[0][2],
                             box2d[0][3],box[5],box[4],box[3],box[0],box[1],box[2],box[6],score),file = f)
    return proc_time


def tracking_val_seq(arg):

    yaml_file = arg.cfg_file

    config = cfg_from_yaml_file(yaml_file,cfg)

    print("\nconfig file:", yaml_file)
    print("data path: ", config.dataset_path)
    print('detections path: ', config.detections_path)

    save_path = config.save_path                       # the results saving path

    os.makedirs(save_path,exist_ok=True)

    seq_list = config.tracking_seqs    # the tracking sequences

    print("tracking seqs: ", seq_list)

    all_time,frame_num = 0,0

    for id in tqdm.trange(len(seq_list)):
        seq_id = seq_list[id]
        before_alloc = torch.cuda.memory_allocated()
        dataset,tracker, this_time, this_num = track_one_seq(seq_id,config)
        after_alloc = torch.cuda.memory_allocated()
        print("Memory used:", after_alloc - before_alloc, "bytes")
        proc_time = save_one_seq(dataset,seq_id,tracker,config)
        torch.cuda.empty_cache()



        all_time+=this_time
        all_time+=proc_time
        frame_num+=this_num

    print("Tracking time: ", all_time)
    print("Tracking frames: ", frame_num)
    print("Tracking FPS:", frame_num/all_time)
    print("Tracking ms:", all_time/frame_num)

    eval_kitti()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default="",
                        help='specify the config for tracking')
    args = parser.parse_args()
    tracking_val_seq(args)

