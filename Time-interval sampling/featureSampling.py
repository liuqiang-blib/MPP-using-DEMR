
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import gc
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

import time





if __name__ == "__main__":
    #
    task_list = ['NR_AR','NR_AR_LBD', 'NR_AhR', 'NR_Aromatase', 'NR_ER', 'NR_ER_LBD','NR_PPAR_gamma', 'SR_ARE', 'SR_ATAD5', 'SR_HSE', 'SR_mmp', 'SR_p53','ADA17','EGFR','HIVPR']
    for  task in task_list:


        df = pd.read_csv("/data/features/" +  task + "_molecule_metadata.csv")
        features = np.load("/data/features/" +  task + "_molecule_features.npy", mmap_mode='r')
        freq_list = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        for freq in freq_list:

            features_sampling =  features[:, ::freq]  # shape: (num_samples, 101, 3, n, n),freq = 100

            df.to_csv("/data/features_sampling_freq_" + str(freq) + '/' +  task + "_molecule_metadata.csv", index=False)
            np.save("/data/features_sampling_freq_"  + str(freq) + '/' +  task + "_molecule_features.npy", features_sampling )

            del features_sampling
            gc.collect()


