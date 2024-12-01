import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, eeg_matrix):
        self.eeg = torch.tensor(eeg_matrix, dtype=torch.float32)

    def __len__(self):
        return self.eeg.shape[0]

    def __getitem__(self, idx):
        return self.eeg[idx].unsqueeze(0), 0