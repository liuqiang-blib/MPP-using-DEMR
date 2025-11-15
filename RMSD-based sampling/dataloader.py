import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
import pandas as pd

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class MoleculeDataset(Dataset):
    def __init__(self, features, labels,mol_id_list, rmsd_dir,mode="low", frame_count=1000):
        assert mode in ["low", "mid", "high"], "mode must be one of ['low', 'mid', 'high']"
        self.features = features
        self.labels = labels
        self.rmsd_dir = rmsd_dir
        self.mol_ids = mol_id_list
        self.mode = mode
        self.frame_count = frame_count

    def __len__(self):
        return len(self.labels)

    def _select_indices(self, rmsd_values):
        sorted_indices = np.argsort(rmsd_values)
        T = len(rmsd_values)
        half = self.frame_count // 2

        if self.mode == "low":
            return sorted_indices[:self.frame_count]
        elif self.mode == "high":
            return sorted_indices[-self.frame_count:]
        elif self.mode == "mid":
            start = max((T // 2) - half, 0)
            end = start + self.frame_count
            return sorted_indices[start:end]

    def __getitem__(self, idx):
        mol_id = self.mol_ids[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        feature = torch.tensor(self.features[idx], dtype=torch.float32)

        # 读取 RMSD 并选帧
        rmsd_path = os.path.join(self.rmsd_dir, f"{mol_id}_rmsd.csv")
        rmsd = pd.read_csv(rmsd_path).iloc[:, 1].values  

        if len(rmsd) < self.frame_count:
            raise ValueError(f"RMSD too short for {mol_id}: only {len(rmsd)} frames")
        indices = self._select_indices(rmsd)
        feature = feature[indices]
        rmsd_selected = torch.tensor(rmsd[indices], dtype=torch.float32)

        # RMSD 归一化权重
        rmsd_min = rmsd_selected.min()
        rmsd_max = rmsd_selected.max()
        rmsd_norm = (rmsd_selected - rmsd_min) / (rmsd_max - rmsd_min + 1e-8)

        return feature, label,rmsd_norm

class MoleculeDataloader:
    def __init__(self, features, labels, mol_id_list, rmsd_dir,mode="low", frame_count=1000, batch_size=32,num_workers=0,seed=42):
    # def __init__(self, features, labels, batch_size=32, num_workers=0,sampler = None):
    #   self.sampler = sampler
        self.batch_size = batch_size
        self.features = features
        self.labels = labels
        self.num_workers = num_workers
        self.molecule_ids = mol_id_list
        self.rmsd_dir = rmsd_dir
        self.mode = mode
        self. frame_count = frame_count

        self.seed = seed
        set_seed(self.seed)
        # self._set_random_seed(self.seed)
        self._split_train_val_test()

    # def _set_random_seed(self, seed):
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     random.seed(seed)

    def _stratified_split(self,train_size=0.7, val_size=0.1, test_size=0.2):

        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, 
        X_trainval, X_test, y_trainval, y_test, id_trainval, id_test= train_test_split(
            self.features, self.labels, self.molecule_ids,
            test_size=test_size,
            random_state=self.seed,
            stratify=self.labels
        )
        relative_val_size = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
            X_trainval, y_trainval, id_trainval,
            test_size=relative_val_size,
            random_state=self.seed,
            stratify=y_trainval
        )
        return X_train, X_val, X_test, y_train, y_val, y_test,id_train, id_val, id_test

    def _split_train_val_test(self):

        (self.train_features, self.val_features, \
            self.test_features, self.train_labels, \
         self.val_labels, self.test_labels,self.train_ids, self.val_ids, self.test_ids ) = self._stratified_split(
            train_size=0.7, val_size=0.1, test_size=0.2,
        )


    def _create_dataloader(self, features, labels, mol_ids, shuffle):
        dataset = MoleculeDataset(features, labels, mol_ids, self.rmsd_dir, self.mode, self.frame_count)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

        # return DataLoader(
        #     dataset,
        #     batch_size=self.batch_size,
        #     shuffle=shuffle,
        #     num_workers=self.num_workers,
        #     sampler = self.sampler
        # )

    def get_train(self, shuffle=True):
        return self._create_dataloader(self.train_features, self.train_labels, self.train_ids, True)
        # return self._create_dataloader(self.train_features, self.train_labels, False, self.sampler)

    def get_val(self):
        return self._create_dataloader(self.val_features,  self.val_labels,self.val_ids,True)
        # return self._create_dataloader(self.val_features, self.val_labels, True,None)

    def get_test(self):
        return self._create_dataloader(self.test_features,  self.test_labels,self.test_ids,True)
        # return self._create_dataloader(self.test_features, self.test_labels, True)


