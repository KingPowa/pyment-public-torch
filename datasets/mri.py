import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms as tr

from typing import Optional, Collection

from .file_based import MedicalDataset
from utils.etc import is_iterable
    
class MRIDataset(Dataset):

    def __init__(self, 
                 dataset: MedicalDataset | Collection[MedicalDataset], 
                 age_range = None,
                 transforms: Optional[Collection[torch.nn.Module]] | Optional[torch.nn.Module] = []):
        if age_range:
            assert is_iterable(age_range) and len(age_range) >= 2
            self.age_range = age_range[:2] if age_range[0] < age_range[1] else age_range[:2:-1]
        else: self.age_range = None

        self.age_range = None
        
        self.dataset = dataset if not is_iterable(dataset, cls=MedicalDataset) else ConcatDataset(dataset)
        self.length = len(self.dataset)
        self.mtransforms = tr.Compose(transforms) if transforms else torch.nn.Identity()

    def __len__(self):
        return self.length
    
    def get_sample(self, index: int):
        slice, age, sex = self.dataset[index]
        # Slice is Modalities x Channels x Width X Length. We add 1 x W x L to Channels
        # Normalize the integer value
        if self.age_range:
            normalized_age = (age - self.age_range[0]) / (self.age_range[1] - self.age_range[0])
        else: 
            normalized_age = age
        # Convert gender to binary (0 for M, 1 for F)
        sex_binary = 0 if sex == 'M' else 1
        return slice, torch.tensor([normalized_age, sex_binary])

    def __getitem__(self, index: int):
        image, cond = self.get_sample(index)
        image = torch.from_numpy(image).float()
        image = self.mtransforms(image)

        return image.unsqueeze(0), cond # Add channel information
