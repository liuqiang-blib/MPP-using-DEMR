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



mol_id = 'activemol7'
zip_path = "./activemol7_output_center.zip"
import os

extract_dir = os.getcwd()
import zipfile

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)



pdb_file ="./activemol7_output_center.pdb"
xtc_file = "./activemol7_md_center.xtc"
print(mol_id )


u = mda.Universe(pdb_file, xtc_file)

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
df.to_csv("./activemol7_rmsd.csv", index=False)


