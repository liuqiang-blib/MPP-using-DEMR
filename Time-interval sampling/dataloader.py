import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class MoleculeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features  
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

class MoleculeDataloader:
    def __init__(self, features, labels, batch_size=32,num_workers=0,seed=42):

        self.batch_size = batch_size
        self.features = features
        self.labels = labels
        self.num_workers = num_workers

        self.seed = seed
        set_seed(self.seed)
        # self._set_random_seed(self.seed)
        self._split_train_val_test()



    def _stratified_split(self,train_size=0.7, val_size=0.1, test_size=0.2):

        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, 
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            self.features, self.labels,
            test_size=test_size,
            random_state=self.seed,
            stratify=self.labels
        )
        relative_val_size = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=relative_val_size,
            random_state=self.seed,
            stratify=y_trainval
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _split_train_val_test(self):

        self.train_features, self.val_features, self.test_features, self.train_labels, self.val_labels, self.test_labels = self._stratified_split(
            train_size=0.7, val_size=0.15, test_size=0.15,
        )


    def _create_dataloader(self, features, labels, shuffle):
        dataset = MoleculeDataset(features, labels)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )



    def get_train(self, shuffle=True):
        return self._create_dataloader(self.train_features, self.train_labels,  True)


    def get_val(self):
        return self._create_dataloader(self.val_features,  self.val_labels,True)


    def get_test(self):
        return self._create_dataloader(self.test_features,  self.test_labels,True)


