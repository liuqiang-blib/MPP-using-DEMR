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
    def __init__(self, features, labels, mol_id_list, rmsd_dir, extract_config={"high": 2, "mid": 2, "low": 2}):
        self.features = features
        self.labels = labels
        self.rmsd_dir = rmsd_dir
        self.mol_ids = mol_id_list
        self.extract_config = extract_config

    def __len__(self):
        return len(self.labels)

    def _select_indices(self, rmsd_values, mode, count):
        sorted_indices = np.argsort(rmsd_values)
        T = len(rmsd_values)
        if count > T:
            raise ValueError(f"Requested {count} frames, but only {T} available.")

        if mode == "low":
            return sorted_indices[:count]
        elif mode == "high":
            return sorted_indices[-count:]
        elif mode == "mid":
            mid_start = max((T - count) // 2, 0)
            return sorted_indices[mid_start:mid_start + count]
        else:
            raise ValueError("Mode must be one of ['low', 'mid', 'high']")

    def __getitem__(self, idx):
        mol_id = self.mol_ids[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        feature = torch.tensor(self.features[idx], dtype=torch.float32)

        rmsd_path = os.path.join(self.rmsd_dir, f"{mol_id}_rmsd.csv")
        rmsd = pd.read_csv(rmsd_path).iloc[:, 1].values

        selected_indices = []
        for mode in ["high", "mid", "low"]:
            count = self.extract_config.get(mode, 0)
            if count > 0:
                indices = self._select_indices(rmsd, mode, count)
                selected_indices.extend(indices)

        selected_indices = np.array(selected_indices)
        feature_selected = feature[selected_indices]
        rmsd_selected = torch.tensor(rmsd[selected_indices], dtype=torch.float32)
        return feature_selected, label, rmsd_selected

class MoleculeDataloader:
    def __init__(self, features, labels, mol_id_list, rmsd_dir, extract_config={"high": 2, "mid": 2, "low": 2},
                 batch_size=32, num_workers=0, seed=42):
        self.batch_size = batch_size
        self.features = features
        self.labels = labels
        self.num_workers = num_workers
        self.molecule_ids = mol_id_list
        self.rmsd_dir = rmsd_dir
        self.extract_config = extract_config
        self.seed = seed


        set_seed(self.seed)
        # self._set_random_seed(self.seed)
        self._split_train_val_test()

    # def _set_random_seed(self, seed):
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     random.seed(seed)

    def _stratified_split(self, train_size=0.7, val_size=0.1, test_size=0.2):
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6,
        X_trainval, X_test, y_trainval, y_test, id_trainval, id_test = train_test_split(
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
        return X_train, X_val, X_test, y_train, y_val, y_test, id_train, id_val, id_test

    def _split_train_val_test(self):
        (self.train_features, self.val_features, self.test_features,
         self.train_labels, self.val_labels, self.test_labels,
         self.train_ids, self.val_ids, self.test_ids) = self._stratified_split(
            train_size=0.7, val_size=0.1, test_size=0.2,
        )

    def _create_dataloader(self, features, labels, mol_ids, shuffle):
        dataset = MoleculeDataset(features, labels, mol_ids, self.rmsd_dir, self.extract_config)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def get_train(self, shuffle=True):
        return self._create_dataloader(self.train_features, self.train_labels, self.train_ids, shuffle)

    def get_val(self):
        return self._create_dataloader(self.val_features, self.val_labels, self.val_ids, True)

    def get_test(self):
        return self._create_dataloader(self.test_features, self.test_labels, self.test_ids, True)


