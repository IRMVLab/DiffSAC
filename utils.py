import torch
import numpy as np
import random
import scipy.optimize
import sklearn.metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def calc_auc(error_array, cutoff=10.):
    error_array = error_array.squeeze()
    error_array = np.sort(error_array)
    num_values = error_array.shape[0]
    
    if num_values == 0:
        return 0, np.array([])
    
    plot_points = np.zeros((num_values, 2))
    midfraction = 1.
    
    for i in range(num_values):
        fraction = (i + 1) / num_values
        value = error_array[i]
        plot_points[i, 1] = fraction
        plot_points[i, 0] = value
        
        if i > 0:
            lastvalue = error_array[i - 1]
            if lastvalue < cutoff < value:
                midfraction = (lastvalue * plot_points[i - 1, 1] + value * fraction) / (value + lastvalue)
    
    if plot_points[-1, 0] < cutoff:
        plot_points = np.vstack([plot_points, np.array([cutoff, 1])])
    else:
        plot_points = np.vstack([plot_points, np.array([cutoff, midfraction])])
    
    sorting = np.argsort(plot_points[:, 0])
    plot_points = plot_points[sorting, :]
    
    area0 = plot_points[plot_points[:, 0] <= cutoff, 0]
    area1 = plot_points[plot_points[:, 0] <= cutoff, 1]
    
    if area0.shape[0] < 2:
        auc = 0
    else:
        auc = sklearn.metrics.auc(area0, area1)
    
    return auc / cutoff, plot_points


def line_2pt_similarity(lines1, lines2):
    if lines1.numel() == 0 or lines2.numel() == 0:
        return torch.tensor([]).to(lines1.device)
    
    lines1 = lines1.unsqueeze(1)
    lines2 = lines2.unsqueeze(0)
    
    dist1 = torch.sqrt(torch.sum((lines1[:, :, :2] - lines2[:, :, :2]) ** 2, dim=2)) + \
            torch.sqrt(torch.sum((lines1[:, :, 2:] - lines2[:, :, 2:]) ** 2, dim=2))
    
    dist2 = torch.sqrt(torch.sum((lines1[:, :, :2] - lines2[:, :, 2:]) ** 2, dim=2)) + \
            torch.sqrt(torch.sum((lines1[:, :, 2:] - lines2[:, :, :2]) ** 2, dim=2))
    
    dist = torch.min(dist1, dist2)
    
    max_dist = 2 * np.sqrt(2)
    similarity = (max_dist - dist) / max_dist
    similarity = torch.clamp(similarity, 0, 1)
    
    return similarity


def single_eval_line(gt_lines, pred_lines):
    if gt_lines.numel() == 0 and pred_lines.numel() == 0:
        return [], 0, [], []
    
    if pred_lines.numel() == 0:
        return [1.0] * gt_lines.size(0), gt_lines.size(0), [], []
    
    if gt_lines.numel() == 0:
        return [], 0, [], []
    
    cost_matrix = line_2pt_similarity(gt_lines, pred_lines)
    cost_matrix = 1.0 - cost_matrix
    cost_matrix = cost_matrix.cpu().numpy()
    
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    
    errors = []
    for ri, ci in zip(row_ind, col_ind):
        errors.append(cost_matrix[ri, ci])
    
    missing_lines = max(0, gt_lines.size(0) - pred_lines.size(0))
    errors.extend([1.0] * missing_lines)
    
    return errors, missing_lines, row_ind, col_ind


def eval_line(total_gt_lines, total_pred_lines):
    all_errors = []
    
    for gt_lines, pred_lines in zip(total_gt_lines, total_pred_lines):
        errors, _, _, _ = single_eval_line(gt_lines, pred_lines)
        all_errors.extend(errors)
    
    if not all_errors:
        return 0.0
    
    auc_05, _ = calc_auc(np.array(all_errors), cutoff=0.5)
    return auc_05 * 100