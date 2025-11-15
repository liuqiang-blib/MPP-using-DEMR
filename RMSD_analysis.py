import MDAnalysis as mda
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def clean_mol_id_list(folder_path):
    cleaned_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith('_output_center.pdb'):

            cleaned_name = filename.replace('_output_center.pdb', '')
            cleaned_names.append(cleaned_name)
    return cleaned_names



task_list = ['TOX21','EGFR','HIVPR','ADA17']

for task in task_list:

    if task == 'TOX21':
        folder_path_pdb = "/data/TOX21/TOX21_Succeed_pdb/"
        folder_path_xtc = "/data/TOX21/TOX21_Succeed_xtc/"
    else:
        folder_path_pdb = "/data/"+ task + "_success/pdb/"
        folder_path_xtc = "/data/" + task + "_success/xtc/"

    mol_id_list =  clean_mol_id_list(folder_path_pdb)

    for mol_id in mol_id_list:

        pdb_file = folder_path_pdb + mol_id + "_output_center.pdb"    # 参考结构
        xtc_file = folder_path_xtc + mol_id + "_md_center.xtc"   # 模拟轨迹文件
        print(mol_id )
        # === 加载轨迹 ===
        u = mda.Universe(pdb_file)

        mobile_atoms = u.select_atoms("all")

        ref = mda.Universe(pdb_file)
        ref_atoms = ref.select_atoms("all")


        R = rms.RMSD(mobile_atoms, ref_atoms)
        R.run()


        rmsd = R.rmsd  # shape: (n_frames, 3) → columns: frame, time (ps), RMSD (Å)
        time_ns = rmsd[:, 1] / 1000
        rmsd_vals = rmsd[:, 2]
        rmsd_vals_nm = rmsd_vals / 10.0


        df = pd.DataFrame({
            'Time (ns)': time_ns,
            'RMSD (nm)': rmsd_vals_nm
        })
        df.to_csv("/data/MD_analysis/" + task + "/" + mol_id +"_rmsd.csv", index=False)


