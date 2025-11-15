
import MDAnalysis as mda
import numpy as np
import pandas as pd
from rdkit import Chem



def get_element_type(element):
    if element in ['F', 'Cl', 'Br', 'I']:
        return 'HAX'
    elif element in ['C', 'N', 'O', 'H', 'P', 'S']:
        return element
    else:
        return 'Du'

def compute_tensor_fingerprint(atom_group, atom_types, trajectory,  bins=[(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]):

    atom_type_to_index = {atom: idx for idx, atom in enumerate(atom_types)}
    N = len(atom_types)  # length of atom-types
    B = len(bins)
    T = len(trajectory)

    # frame F and atom number M
    fingerprint_tensor = np.zeros((N, N, B, T), dtype=int)  # 初始化张量

    # preprocess atom type
    atom_elements = [atom.element for atom in atom_group]

    converted_atom_types = [get_element_type(el) for el in atom_elements]
    atom_type_indices = [atom_type_to_index.get(el, -1) for el in converted_atom_types]

    for frame_idx, ts in enumerate(trajectory):
        atom_group.positions = ts.positions
        positions = atom_group.positions

        # calculate distance matrix
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))

        # Create a mask matrix that identifies pairs of atoms that meet a distance threshold
        for bin_idx, (low, high) in enumerate(bins):
            contact_mask = (dist_matrix < high) & (dist_matrix >= low)


        # Iterate over the entire matrix and count the statistics into tensors
            for i in range(len(converted_atom_types )):
                type_i = atom_type_indices[i]
                if type_i == -1:  # Skip unknown atom types
                    continue

                for j in range(len(converted_atom_types )):  # Iterate over all atoms
                    if i == j:  # skip over oneself
                        continue

                    type_j = atom_type_indices[j]
                    if type_j == -1:
                        continue

                    if contact_mask[i, j]:
                        fingerprint_tensor[type_i, type_j, bin_idx, frame_idx] += 1
    return fingerprint_tensor






def atom_type(smiles_list):
    all_atoms = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            for atom in mol.GetAtoms():
                all_atoms.add(atom.GetSymbol())  
        else:
            print(f"Invalid SMILES: {smi}")
    return all_atoms


if __name__ == "__main__":


    atom_types = ['C', 'N', 'O', 'H', 'P', 'S', 'HAX', 'Du']


    valid_indices = []
    feature_list = []
    bins = [(1.0, 2.0), (2.0, 3.0), (3.0, 4.0)]

    mol_id = 'activemol7'

    zip_path = "./activemol7_output_center.zip"
    import os
    extract_dir = os.getcwd()  # 获取当前工作目录
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)




    pdb_file = './activemol7_output_center.pdb'
    xtc_file = './activemol7_md_center.xtc'


    u = mda.Universe(pdb_file,xtc_file)
    trajectory = u.trajectory

    if trajectory.n_frames != 10001:
        print(f"skipping {mol_id}，Frame {trajectory.n_frames}")

    atom_group = u.atoms
    #
    fingerprint_tensor = compute_tensor_fingerprint(atom_group, atom_types, trajectory, bins)

    fingerprint_tensor = fingerprint_tensor.transpose(3, 2, 0, 1)  # (T, B, N, N)

    # print(f"Number {len(feature_list) + 1} | mol_id: {mol_id} | Fingerprint shape: {fingerprint_tensor.shape}")

    feature_list.append(fingerprint_tensor)




    features_array = np.stack(feature_list)

    np.save("./activate7_molecule_features.npy", features_array)



