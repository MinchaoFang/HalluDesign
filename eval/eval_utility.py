import numpy as np
import string
from rdkit import Chem
from rdkit.Chem import AllChem
import random
from Bio import PDB

def generate_metrics(num_protein,num_small_molecular, num_dna, num_rna,chain_types):

    metrics = {
        'file_name': None ,
        'cycle': np.nan,
        "eval_path" : None,
        "packed_path" : None ,
        "oringin_path" : None ,
        "plddt_good_indicates_len" : np.nan ,
        "key_interaction_len": np.nan,
        'op_cif_path' : None ,
        'AF3_Status': 'Not Run',
        "eval_status" : "Not Run",
        "eval_plddt" : np.nan ,
        "op_plddt" : None , 
        'eval_iptm' : np.nan ,
        'eval_ptm'  : np.nan ,
        'op_iptm' : np.nan ,
        'op_ptm'  : np.nan ,
        'op_pae': np.nan,
        'op_pde': np.nan,
        'op_ipae': np.nan,
        'op_ipde': np.nan,
        'mpnn_sequence': None,
        "mpnn_score" : np.nan,
        "prediction_model" : None,
        "mpnn_model" : None,
        'eval_plddt_fix': np.nan,
        'eval_plddt_redes': np.nan,
        'eval_pae': np.nan,
        'eval_pde': np.nan,
        'eval_ipae': np.nan,
        'eval_ipde': np.nan,
        "esm_score": np.nan,
    }
    
    num_chains = num_protein + num_small_molecular + num_dna + num_rna
    chain_labels = string.ascii_uppercase[:num_chains]  
    if num_protein != 0:
        for i in range(0,num_protein):
            label = chain_labels[i]
            print("protein",label)
            metrics[f'eval_{label}_plddt'] = np.nan
            metrics[f'eval_{label}_ptm'] = np.nan
            metrics[f'op_{label}_ptm'] = np.nan
            metrics[f'op_{label}_plddt'] = np.nan

            
    if num_small_molecular != 0 or num_dna != 0 or num_rna != 0 :
        metrics[f'eval_all_ipae_to_protein'] = np.nan
        metrics[f'eval_all_iptm_to_protein'] = np.nan
        metrics[f'eval_key_res_plddt'] = np.nan
        metrics[f'op_all_ipae_to_protein'] = np.nan
        metrics[f'op_all_iptm_to_protein'] = np.nan
        metrics[f'op_key_res_plddt'] = np.nan
        for j in range(num_protein,num_chains):
            label = chain_labels[j]
            metrics[f'eval_{label}_plddt'] = np.nan
            metrics[f'eval_{label}_ptm'] = np.nan
            metrics[f'eval_{label}_iptm'] = np.nan
            metrics[f'eval_{label}_ipae'] = np.nan
            metrics[f'op_{label}_plddt'] = np.nan
            metrics[f'op_{label}_ptm'] = np.nan
            metrics[f'op_{label}_iptm'] = np.nan
            metrics[f'op_{label}_ipae'] = np.nan
    count =0
    
    for chain in chain_types:
        label = chain_labels[count]
        if chain == "protein":
            metrics[f'eval_protein_{label}_rmsd'] = np.nan
            metrics[f'op_protein_{label}_rmsd'] = np.nan
            metrics[f'origin_protein_{label}_rmsd'] = np.nan
        if chain == 'ligand':
            metrics[f'eval_ligand_{label}_rmsd'] = np.nan
            metrics[f'eval_atom_distances_{label}'] = np.nan
            metrics[f'op_ligand_{label}_rmsd'] = np.nan
            metrics[f'op_atom_distances_{label}'] = np.nan
            metrics[f'origin_ligand_{label}_rmsd'] = np.nan
            metrics[f'origin_atom_distances_{label}'] = np.nan
        if chain == 'dna':
            metrics[f'op_dna_{label}_rmsd'] = np.nan
            metrics[f'origin_dna_{label}_rmsd'] = np.nan
            metrics[f'eval_dna_{label}_rmsd'] = np.nan
        if chain == 'rna':
            metrics[f'eval_rna_{label}_rmsd'] = np.nan
            metrics[f'op_rna_{label}_rmsd'] = np.nan
            metrics[f'origin_rna_{label}_rmsd'] = np.nan
        count +=1
    print(metrics)

    return metrics


def get_random_seeds(num_seeds: int):
    import random
    random_numbers = [random.randint(0, 10000) for _ in range(num_seeds)]
    
    return random_numbers

import os

from rdkit import Chem

from rdkit.Chem import AllChem

from Bio import PDB

import numpy as np

from scipy.spatial.transform import Rotation as R 

def get_geometric_center(atoms):
    coords = np.array([atom.coord for atom in atoms])
    center = np.mean(coords, axis=0)
    return center

def get_b_chain_center(structure):
    """Get the geometric center of atoms in the B chain."""
    for model in structure:
        for chain in model:
            if chain.id == 'B':
                atoms = [atom for atom in chain.get_atoms()]
                return get_geometric_center(atoms)
    return None

def apply_random_rotation(coords):
    r = R.random()  # Generate random rotation

    rotated_coords = r.apply(coords)
    return rotated_coords

def place_molecule_at_center(molecule, center):
    conformer = molecule.GetConformer()
    mol_coords = np.array([conformer.GetAtomPosition(i) for i in range(molecule.GetNumAtoms())])
    
    # Apply random rotation

    rotated_coords = apply_random_rotation(mol_coords)
    
    mol_center = np.mean(rotated_coords, axis=0)
    translation_vector = center - mol_center

    for i in range(molecule.GetNumAtoms()):
        new_pos = rotated_coords[i] + translation_vector

        conformer.SetAtomPosition(i, new_pos)

def recentre_pdb(input_file, output_file, smiles):
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule)
    AllChem.UFFOptimizeMolecule(molecule)

    parser = PDB.MMCIFParser(QUIET=True)
    io = PDB.PDBIO()

    filename = os.path.basename(input_file)
    structure = parser.get_structure(filename, input_file)
    
    # Get the geometric center of the small molecule in chain B
    center = get_b_chain_center(structure)
    if center is None:
        print(f"No B chain found in {input_file}. Skipping.")
    
    # Place the molecule at the geometric center of chain B
    place_molecule_at_center(molecule, center)
    # Keep only chain A
    model = structure[0]
    chains_to_remove = [chain for chain in model if chain.id != 'A']
    for chain in chains_to_remove:
        model.detach_child(chain.id)
    # Add new chain B
    new_chain_id = 'B'
    if new_chain_id in model:
        raise ValueError(f"Chain ID {new_chain_id} already exists. Choose a different ID.")
    
    new_chain = PDB.Chain.Chain(new_chain_id)
    model.add(new_chain)
    new_residue = PDB.Residue.Residue(('H_MOL', 1, ' '), 'MOL', ' ')
    new_chain.add(new_residue)
    for idx, atom in enumerate(molecule.GetAtoms()):
        if atom.GetSymbol() != 'H':  # Skip H atom
            pos = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
            atom_name = f"{atom.GetSymbol()}{idx + 1}".ljust(4)
            new_atom = PDB.Atom.Atom(
                atom_name,
                np.array([pos.x, pos.y, pos.z]),
                0.0,
                1.0,
                ' ',
                atom_name,
                idx + 1,
                element=atom.GetSymbol()
            )
            new_residue.add(new_atom)
    io.set_structure(structure)
    io.save(output_file)
    


import subprocess

def get_gpu_memory():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        # Output format example: '44588, 16894, 39694\n'
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        for i, line in enumerate(lines):
            total, used, free = line.split(',')
            gpu_info.append({
                'gpu_index': i,
                'memory_total_MB': int(total.strip()),
                'memory_used_MB': int(used.strip()),
                'memory_free_MB': int(free.strip())
            })
        return gpu_info
    except Exception as e:
        print("Error running nvidia-smi:", e)
        return None

class PeptideSynthesizer:
    def __init__(self):
        pass

    def build_residue(self, aa):
        aa_smiles = {
        'ALA': 'N[C@@H](C)C(=O)O',
        'GLY': 'NCC(=O)O',
        'SER': 'N[C@@H](CO)C(=O)O',
        'ARG': 'N[C@@H](CCCNC(=N)N)C(=O)O',
        'ASN': 'N[C@@H](CC(=O)N)C(=O)O',
        'ASP': 'C([C@@H](C(=O)O)N)C(=O)O',
        'CYS': 'N[C@@H](CS)C(=O)O',
        'GLN': 'N[C@@H](CCC(=O)N)C(=O)O',
        'GLU': 'N[C@@H](CCC(=O)O)C(=O)O',
        'HIS': 'N[C@@H](CC1=CN=CN1)C(=O)O',
        'ILE': 'N[C@@H](C(C)CC)C(=O)O',
        'LEU': 'N[C@@H](CC(C)C)C(=O)O',
        'LYS': 'N[C@@H](CCCCN)C(=O)O',
        'MET': 'N[C@@H](CCSC)C(=O)O',
        'PHE': 'N[C@@H](CC1=CC=CC=C1)C(=O)O',
        'PRO': 'N1CCC[C@H]1C(=O)O',
        'THR': 'N[C@@H](C(O)C)C(=O)O',
        'TRP': 'N[C@@H](CC1=CNC2=CC=CC=C12)C(=O)O',
        'TYR': 'N[C@@H](CC1=CC=C(O)C=C1)C(=O)O',
        'VAL': 'N[C@@H](C(C)C)C(=O)O',
    }
        mol = Chem.MolFromSmiles(aa_smiles[aa])
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol)
        return mol

    def connect_residues(self, mol1, mol2):
        c_carbons = [atom.GetIdx() for atom in mol1.GetAtoms() if atom.GetAtomicNum() == 6 and
                     any(nbr.GetAtomicNum() == 8 and mol1.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE for nbr in atom.GetNeighbors())]
        c_idx = c_carbons[-1]

        n_atoms = [atom.GetIdx() for atom in mol2.GetAtoms() if atom.GetAtomicNum() == 7 and
                   any(nbr.GetAtomicNum() == 1 for nbr in atom.GetNeighbors())]
        n_idx = n_atoms[0]

        combo = Chem.CombineMols(mol1, mol2)
        rw_mol = Chem.RWMol(combo)
        offset = mol1.GetNumAtoms()
        n_idx += offset

        # Remove -OH from carboxyl group
        oxy_idx, hyd_idx = None, None
        for nbr in rw_mol.GetAtomWithIdx(c_idx).GetNeighbors():
            a = rw_mol.GetAtomWithIdx(nbr.GetIdx())
            if a.GetAtomicNum() == 8 and rw_mol.GetBondBetweenAtoms(c_idx, a.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                hs = [x.GetIdx() for x in a.GetNeighbors() if rw_mol.GetAtomWithIdx(x.GetIdx()).GetAtomicNum() == 1]
                if len(hs) == 1:
                    oxy_idx = a.GetIdx()
                    hyd_idx = hs[0]
                    break
        if oxy_idx is None or hyd_idx is None:
            raise RuntimeError("Could not find carboxyl -OH")

        for idx in sorted([hyd_idx, oxy_idx], reverse=True):
            rw_mol.RemoveAtom(idx)
            if idx < n_idx:
                n_idx -= 1

        h_on_n = [a.GetIdx() for a in rw_mol.GetAtomWithIdx(n_idx).GetNeighbors() if rw_mol.GetAtomWithIdx(a.GetIdx()).GetAtomicNum() == 1]
        if not h_on_n:
            raise RuntimeError("Could not find H on N-terminal amino group")
        rw_mol.RemoveAtom(h_on_n[0])
        if h_on_n[0] < n_idx:
            n_idx -= 1

        rw_mol.AddBond(c_idx, n_idx, Chem.BondType.SINGLE)

        mol = rw_mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol

    def cyclize(self, mol):
        rw_mol = Chem.RWMol(mol)

        def find_carbons(mol):
            return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6 and
                    any(nbr.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                        for nbr in atom.GetNeighbors())]

        c_carbons = find_carbons(rw_mol)
        if not c_carbons:
            raise RuntimeError("Could not find carboxyl carbon")
        c_idx = c_carbons[-1]

        oxy_idx, hyd_idx = None, None
        for nbr in rw_mol.GetAtomWithIdx(c_idx).GetNeighbors():
            atom = rw_mol.GetAtomWithIdx(nbr.GetIdx())
            if atom.GetAtomicNum() == 8 and rw_mol.GetBondBetweenAtoms(c_idx, atom.GetIdx()).GetBondType() == Chem.BondType.SINGLE:
                hs = [h.GetIdx() for h in atom.GetNeighbors() if rw_mol.GetAtomWithIdx(h.GetIdx()).GetAtomicNum() == 1]
                if len(hs) == 1:
                    oxy_idx = atom.GetIdx()
                    hyd_idx = hs[0]
                    break
        if oxy_idx is None or hyd_idx is None:
            raise RuntimeError("Could not find hydroxyl O-H")

        # Find N-terminal amino N
        n_atoms = [a.GetIdx() for a in rw_mol.GetAtoms() if a.GetAtomicNum() == 7 and
                   any(n.GetAtomicNum() == 1 for n in a.GetNeighbors())]
        if not n_atoms:
            raise RuntimeError("Could not find N-terminal amino N")
        n_idx = n_atoms[0]

        # Remove hydroxyl group
        for idx in sorted([hyd_idx, oxy_idx], reverse=True):
            rw_mol.RemoveAtom(idx)
            if idx < c_idx:
                c_idx -= 1
            if idx < n_idx:
                n_idx -= 1

        # Remove N-terminal hydrogen
        h_on_n = [a.GetIdx() for a in rw_mol.GetAtomWithIdx(n_idx).GetNeighbors() if rw_mol.GetAtomWithIdx(a.GetIdx()).GetAtomicNum() == 1]
        if not h_on_n:
            raise RuntimeError("Could not find N-terminal hydrogen")
        rw_mol.RemoveAtom(h_on_n[0])
        if h_on_n[0] < c_idx:
            c_idx -= 1
        if h_on_n[0] < n_idx:
            n_idx -= 1

        rw_mol.AddBond(c_idx, n_idx, Chem.BondType.SINGLE)

        mol = rw_mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol

    def sequence_to_peptide_smiles(self, sequence, cyclic=False):
        mol = self.build_residue(sequence[0])
        for aa in sequence[1:]:
            next_mol = self.build_residue(aa)
            mol = self.connect_residues(mol, next_mol)

        smiles_linear = Chem.MolToSmiles(mol, isomericSmiles=True)

        if cyclic:
            mol = self.cyclize(mol)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        else:
            smiles = smiles_linear
        mol = Chem.MolFromSmiles(smiles)
        Chem.rdDepictor.Compute2DCoords(mol)
        standard_smiles = Chem.MolToSmiles(mol, canonical=True)
        return standard_smiles
    
    def convert_single_to_three_letter_sequence(self, single_letter_seq):
        aa_map = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        three_letter_seq = []
        for aa in single_letter_seq:
            if aa in aa_map:
                three_letter_seq.append(aa_map[aa])
            else:
                raise ValueError(f"Invalid amino acid: {aa}")
        return three_letter_seq
    
from Bio.PDB import PDBParser
def get_global_residue_index(pdb_path, residue_code):
    """
    Given a residue code (e.g., B1), return its global index across all chains 
    (counting sequentially starting from chain A).
    
    :param pdb_path: Path to the PDB file.
    :param residue_code: Residue code, such as "B1".
    :return: Global index (counting starts from chain A), and corresponding chain and residue information.
    """
    parser = PDBParser(QUIET=True)  

    structure = parser.get_structure("PDB_structure", pdb_path)

    # Parse residue code: e.g., "B1" is separated into chain ID "B" and residue number 1

    chain_id = residue_code[0]  # Chain ID part
    print(f"chain_id: {chain_id}")

    try:
        residue_id = int(residue_code[1:])  # Residue number part (assuming no insertion code)
    except ValueError:
        raise ValueError(f"Could not parse residue number from {residue_code}!")

    # Iterate over all chains and residues to calculate the global index

    global_index = 0  # Global index, counting starts from 1

    for chain in structure.get_chains():
        chain_name = chain.get_id()
        for residue in chain.get_residues():
            global_index += 1  # Increment global index for each residue encountered

            # Check if the current chain and residue match the specified code
            if chain_name == chain_id and residue.get_id()[1] == residue_id:
                return global_index, chain_name, residue.get_id()

    # Raise exception if no matching residue code is found

    raise ValueError(f"Residue code {residue_code} not found in the PDB file!")