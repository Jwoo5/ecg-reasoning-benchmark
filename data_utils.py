# data_utils.py
import os
import torch
import numpy as np
import wfdb
from abc import ABC, abstractmethod

class BaseECGDataset(ABC):
    def __init__(self, base_dir, seq_length=5000):
        self.base_dir = base_dir
        self.seq_length = seq_length

    def process_signal(self, ecg_data):

        # ecg_data[np.isnan(ecg_data)] = 0
        # ecg_data[np.isinf(ecg_data)] = 0

        ecg = torch.tensor(np.transpose(ecg_data, (1, 0)).astype(np.float32))
        
        c, length = ecg.shape
        if length < self.seq_length:
            new_ecg = torch.zeros((c, self.seq_length))
            new_ecg[:, 0:length] = ecg
            ecg = new_ecg
        elif length > self.seq_length:
            ecg = ecg[:, 0:self.seq_length]
            
        return ecg

    @abstractmethod
    def load_signal(self, ecg_id):
        pass

class PTBXLDataset(BaseECGDataset):
    def load_signal(self, ecg_id):

        ecg_num_int = int(ecg_id)
        dir_num = (ecg_num_int // 1000) * 1000
        

        path = os.path.join(self.base_dir, "records500", f"{dir_num:05d}", f"{ecg_num_int:05d}_hr")
        
        try:
            ecg_data, meta = wfdb.rdsamp(path)
            sr = meta['fs']
        except Exception as e:
            print(f"Error reading PTB-XL file {path}: {e}")
            raise e

        return self.process_signal(ecg_data), sr

class MimicIVDataset(BaseECGDataset):
    def load_signal(self, ecg_id):
        pass

def get_dataset_loader(dataset_name, base_dir):
    """Factory function to switch logic based on dataset name"""
    name = dataset_name.lower()
    if "ptb" in name:
        return PTBXLDataset(base_dir)
    elif "mimic" in name:
        return MimicIVDataset(base_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported in data_utils.py")