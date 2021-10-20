import numpy as np

import os

import torch
from torch.utils.data import Dataset


def read_npz(split):
    data_directory_path = os.path.join("data")  
    train_path = data_directory_path + "/train.npz"
    valid_path = data_directory_path + "/valid.npz"
        
    assert os.path.exists(train_path), "train.npz must exist in the data directory"
    assert os.path.exists(valid_path), "valid.npz must exist in the data directory"
        
    path = train_path if (split == 'train') else valid_path

    data_npz = np.load(path)

    samples = data_npz['samples']
    annotations = data_npz['annotations']
        
    data_npz.close()

    return (samples, annotations)

class Simple2DDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        assert split in ['train', 'valid'], f'Split parameters "{split}" must be either "train" or "valid".'
        # Read either train or validation data from disk based on split parameter using np.load.
        # Data is located in the folder "data".
        # Hint: you can use os.path.join to obtain a path in a subfolder.
        # Save samples and annotations to class members self.samples and self.annotations respectively.
        # Samples should be an Nx2 numpy array. Annotations should be Nx1.
        self.samples, self.annotations = read_npz(split)
        #raise NotImplementedError()
            
    def __len__(self):
        # Returns the number of samples in the dataset.
        return self.samples.shape[0]
    
    def __getitem__(self, idx):
        # Returns the sample and annotation with index idx.
        #raise NotImplementedError()
        sample = self.samples[idx]
        annotation = self.annotations[idx]
        
        # Convert to tensor.
        return {
            'input': torch.from_numpy(sample).float(),
            'annotation': torch.from_numpy(annotation[np.newaxis]).float()
        }


class Simple2DTransformDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        assert split in ['train', 'valid'], f'Split parameters "{split}" must be either "train" or "valid".'
        # Read either train or validation data from disk based on split parameter.
        # Data is located in the folder "data".
        # Hint: you can use os.path.join to obtain a path in a subfolder.
        # Save samples and annotations to class members self.samples and self.annotations respectively.
        # Samples should be an Nx2 numpy array. Annotations should be Nx1.
        
        self.samples, self.annotations = read_npz(split)

        #raise NotImplementedError()
            
    def __len__(self):
        # Returns the number of samples in the dataset.
        return self.samples.shape[0]
    
    def __getitem__(self, idx):
        # Returns the sample and annotation with index idx.
        
        #raise NotImplementedError()
        
        sample = self.samples[idx]
        annotation = self.annotations[idx]
        
        # Transform the sample to a different coordinate system.
        sample = transform(sample)

        # Convert to tensor.
        return {
            'input': torch.from_numpy(sample).float(),
            'annotation': torch.from_numpy(annotation[np.newaxis]).float()
        }


def transform(sample):
    #raise NotImplementedError()
    new_sample = np.array([sample[0], sample[1], sample[0]**2 + sample[1]**2])
    return new_sample
