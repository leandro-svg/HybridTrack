import numpy as np
from navpy import angle2dcm as a2d
from scipy.linalg import logm, svd
import json
def evaluate_rot(pred_euler, gt_euler, n=180, only_verbose=1, details=0):
    # https://github.com/ShapeNet/RenderForCNN/blob/master/view_estimation/compute_recall_precision_accuracy_3dview.m
    gt_R = a2d(gt_euler[:,0], gt_euler[:,1], gt_euler[:,2])
    pred_R = a2d(pred_euler[:,0], pred_euler[:,1], pred_euler[:,2])
    if len(gt_R.shape) == 2:
        gt_R = gt_R[None, :, :]
    if len(pred_R.shape) == 2:
        pred_R = pred_R[None, :, :]
    diff_R = np.matmul(pred_R.transpose(0, 2, 1), gt_R)
    # https://nl.mathworks.com/help/matlab/ref/norm.html#d123e955559 matlab --> numpy
    res = [np.max(svd(logm(diff_R[i,:,:].squeeze()))[1]) / np.sqrt(2) for i in range(diff_R.shape[0])]
    res = np.array(res)
    acc = np.sum(res < np.pi/180 * n) / res.shape[0]
    mederr = np.median(res)
    out_info = '[*] Rotation Evaluation \nAcc(pi/{}) = {:.2f}%, Mederr = {:.2f}(deg)\n'.format(180//n, acc * 100., mederr / np.pi * 180.)
    if only_verbose:
        return out_info
    else:
        if details:
            return acc, mederr, res, out_info
        else:
            return acc, mederr, out_info



def evaluate_heading(pred_heading, gt_heading, n=180, only_verbose=1, details=0):
    """
    Evaluate the accuracy of the predicted heading angle against the ground truth.

    Args:
    pred_heading (np.array): Predicted heading angles.
    gt_heading (np.array): Ground truth heading angles.
    n (int, optional): The factor for accuracy calculation. Default is 180.
    only_verbose (int, optional): Flag to return only verbose output. Default is 1.
    details (int, optional): Flag to return detailed output. Default is 0.

    Returns:
    String or tuple: Evaluation results depending on the flags.
    """

    # Compute the angular difference in radians, ensuring it's within [-pi, pi]
    diff = np.abs(pred_heading - gt_heading)
    diff = np.minimum(diff, 2 * np.pi - diff)

    # Compute accuracy and median error
    acc = np.sum(diff < np.pi/180 * n) / diff.size
    mederr = np.median(diff)

    # Convert median error to degrees
    mederr_deg = mederr * 180 / np.pi

    # Output information
    out_info = '[*] Heading Evaluation \nAcc(pi/{}) = {:.2f}%, Mederr = {:.2f}(deg)\n'.format(
        180//n, acc * 100., mederr_deg)

    if only_verbose:
        return out_info
    else:
        if details:
            return acc, mederr_deg, diff, out_info
        else:
            return acc, mederr_deg, out_info


def evaluate_trans(pred_trans, gt_trans, mode='XYZ', only_verbose=1, details=0, ctype='none'):
    abs_diff = np.abs(gt_trans - pred_trans)
    abs_diff_list = abs_diff.tolist()

    if mode == 'XYZ':
        sq_diff = np.sqrt(np.sum(np.multiply(abs_diff, abs_diff), axis=1))
        sq_diff_list = sq_diff.tolist()

        # Write to JSON
        
        
        gt_sq = np.sqrt(np.sum(np.multiply(gt_trans, gt_trans), axis=1))
        sq_rel_error = np.mean([sq_diff[i] for i in range(gt_sq.shape[0])])

          
        #        json.dump(sq_diff_list, json_file)
        # sq_diff = np.sqrt((np.multiply(abs_diff[:,0], abs_diff[:,0])))
        # gt_sq = np.sqrt((np.multiply(gt_trans[:,0], gt_trans[:,0])))
        # error_x = np.mean([sq_diff[i] / gt_sq[i] for i in range(gt_sq.shape[0])])

        # sq_diff = np.sqrt((np.multiply(abs_diff[:,1], abs_diff[:,1])))
        # gt_sq = np.sqrt((np.multiply(gt_trans[:,1], gt_trans[:,1])))
        # error_y = np.mean([sq_diff[i] / gt_sq[i] for i in range(gt_sq.shape[0])])

        # sq_diff = np.sqrt((np.multiply(abs_diff[:,2], abs_diff[:,2])))
        # gt_sq = np.sqrt((np.multiply(gt_trans[:,2], gt_trans[:,2])))
        # error_z = np.mean([sq_diff[i] / gt_sq[i] for i in range(gt_sq.shape[0])])
        
        error_x = np.mean(abs_diff[:,0])
        error_y = np.mean(abs_diff[:,1] )
        error_z = np.mean(abs_diff[:,2])
        out_info = '[*] Translation Evaluation \nRelative L2 Error = {:.2f}%\n'.format(sq_rel_error )
        out_info += f'[-] [eX, eY, eZ] (%) = [{error_x }, {error_y }, {error_z }] \n'
        if only_verbose:
            return out_info
        else:
            if details:
                return sq_rel_error, np.array([sq_diff[i] / gt_sq[i] for i in range(gt_sq.shape[0])]), error_x, error_y, error_z, out_info
            else:
                return sq_rel_error, error_x, error_y, error_z, out_info
    else:
        sq_diff = np.sqrt(np.multiply(abs_diff, abs_diff))
        gt_sq = np.sqrt(np.multiply(gt_trans, gt_trans))
        sq_rel_error = np.mean([sq_diff[i] / gt_sq[i] for i in range(gt_sq.shape[0])])
        out_info = '[*] Translation Evaluation \nRelative L2 Error = {:.2f}%\n'.format(sq_rel_error * 100)
        if only_verbose:
            return out_info
        else:
            return sq_rel_error * 100, out_info