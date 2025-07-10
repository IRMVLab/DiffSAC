import os
import numpy as np
import torch
from torch.utils.data import Dataset


class Dataset_line(Dataset):
    
    def __init__(self, data_path, data_num='all'):
        self.data_path = data_path
        self.files = os.listdir(os.path.join(data_path, 'data'))
        self.data_num = len(self.files) if data_num == 'all' else data_num
    
    def __getitem__(self, idx):
        filename = self.files[idx]
        data_path = os.path.join(self.data_path, 'data', filename)
        img_path = os.path.join(self.data_path, 'images', filename.replace('.npz', '.jpg'))
        
        data = np.load(data_path, allow_pickle=True)
        GT_lines = data['lines']
        GT_points = data['points']
        
        np.random.shuffle(GT_points)
        points = GT_points[:, :2]
        
        return points.copy(), GT_lines.copy(), data_path, img_path
    
    def __len__(self):
        return self.data_num


def line_dataset_collate(batch):
    points, lines, data_paths, img_paths = [], [], [], []
    
    for point, line, data_path, img_path in batch:
        points.append(torch.from_numpy(point).float())
        lines.append(torch.from_numpy(line).float())
        data_paths.append(data_path)
        img_paths.append(img_path)
    
    return points, lines, data_paths, img_paths