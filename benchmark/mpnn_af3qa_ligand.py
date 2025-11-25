import os
import argparse
import pandas as pd
import sys
from pathlib import Path
import json
import glob
import subprocess
import shutil
from Bio.PDB import MMCIFParser, PDBParser, PDBIO
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
import os
import shutil
from input_pkl_preprocess import process_single_file
import subprocess
import math
import json
import sys
from data_utility import find_pocket_residues , common_elements
sys.path.insert(0,"/storage/caolongxingLab/fangminchao/Proteus/Proteus_flow_matching")
import esm.esmfold.v1.pretrained
import esm.esmfold.v1.esmfold
from copy import deepcopy
from Bio.PDB import PDBParser, MMCIFParser
import json
import csv
import os
from omegaconf import OmegaConf
from colabdesign.af.model import mk_af_model
from ProteinMPNN.protein_mpnn_utils import model_init
from ProteinMPNN.protein_mpnn_pyrosetta import mpnn_design
from data.parsers import from_pdb_string
from scripts.self_consistency_evaluation import run_folding_and_evaluation
from data import protein
from Bio.PDB import MMCIFParser, PDBIO
import torch
import pose_sequence 
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import numpy as np
from Bio.PDB import PDBParser


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process multiple PDB files with AF2 and AF3')
    parser.add_argument('--pdb_list', type=str, required=True,
                       help='Path to text file containing list of PDB files')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing PDB files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory for output files')
    parser.add_argument('--num_seqs', type=int, default=8,
                       help='Number of proteinMPNN seqs to perform self consistency')
    parser.add_argument('--num_recycles', type=int, default=10,
                       help='Number of recycles to perform')
    parser.add_argument('--ref_time_steps', type=int, default=30,
                       help='ref time steps to perform')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='AF3 qa batchsize')
    parser.add_argument('--num_seeds', type=int, default=1,
                       help='AF3 number random seeds')
    parser.add_argument('--early_stop_threshold', type=int, default=100,
                       help='AF3 early_stop_plddt_threshold')
    parser.add_argument('--template_plddt_threshold', type=int, default=0,
                       help='AF3 template_plddt_threshold')
    parser.add_argument('--sm_plddt_threshold', type=int, default=0,
                       help='AF3 sm template_plddt_threshold')
    parser.add_argument('--template_path', type=str, required=True,
                       help='Path to template.json file')
    parser.add_argument('--reconstruct', type=str, required=True,
                       help='pose_seq_scaffold or af2 or none')
    parser.add_argument('--extra_res_params', type=str, required=True,
                       help='extra res params location')
    parser.add_argument('--ddg', type=str, required=True,
                       help='whether to caluate ddg')
    parser.add_argument('--alphafold', type=bool,  default=True,
                       help='AF2 or esmfold')
    parser.add_argument('--ref_eval', type=int, default=None,
                    help='whether to use small molecular ref position in AF3 evaluation')
    parser.add_argument('--fake_msa', type=int,  default=None,
                       help='whether to use how many ProteinMPNN seqs to fake MSA')
    parser.add_argument('--sm', type=str, required=False,
                    help='whether to use small molecular mode, it will use AF3 and LigandMPNN to evaluate')
    return parser.parse_args()


def convert_cif_to_pdb(cif_file, pdb_file):
    """
    Convert a CIF file to PDB format, ensuring correct handling of protein residues
    and small molecules. Non-protein residues are renamed as 'LIG'.

    Args:
        cif_file (str): Path to the input CIF file.
        pdb_file (str): Path to the output PDB file.

    Returns:
        bool: True if conversion is successful, False otherwise.
    """
    
    # Common amino acid residue names
    amino_acids = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", 
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", 
        "TYR", "VAL", "ASX", "GLX", "PYR", "SEC", "SEL", "XLE", "XAA"
    ]
    
    try:
        # Parse CIF file
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("complex", cif_file)
        
        # Iterate over structure and identify proteins vs small molecules
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname()
                    
                    # If residue name is not in the amino acid list, treat as small molecule
                    if resname not in amino_acids:
                        # Rename small molecule residue as 'LIG'
                        print(f"Renaming small molecule {resname} to LIG")
                        residue.resname = "LIG"
        
        # Save structure as PDB
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_file)

        print(f"Conversion successful. Saved as {pdb_file}")
        return True
    except Exception as e:
        print(f"Failed to convert CIF to PDB: {str(e)}")
        return False



def self_consistency_init(alpahfold,num_seqs):
    mpnn_config_dict = {
            "ca_only": True,
            "model_name": "v_48_020",
            'backbone_noise': 0.00,
            'temperature': 0.1,
            'num_seqs': num_seqs,
        }
    mpnn_config_dict = OmegaConf.create(mpnn_config_dict)
    mpnn_model = model_init(mpnn_config_dict, device='cuda')
    cfg = OmegaConf.load('/storage/caolongxingLab/fangminchao/Proteus/Proteus_flow_matching/configs/inference.yaml')
    af2_configs = cfg.inference.self_consistency.structure_prediction.alphafold
    af2_setting = {
        "models": [3] ,
        "num_recycles": af2_configs.num_recycles,
        'prefix': 'monomer',
        'params_dir': f'/storage/caolongxingLab/fangminchao/Proteus/Proteus_flow_matching/{cfg.inference.self_consistency.structure_prediction.alphafold.params_dir}'
    }
    if alpahfold:
        prediction_model = mk_af_model(
                protocol="hallucination", 
                initial_guess=False, 
                use_initial_atom_pos=False, num_recycles=af2_configs.num_recycles, 
                data_dir=af2_setting['params_dir'],
            )
    else:
        prediction_model = esm.pretrained.esmfold_v1()
        prediction_model = prediction_model.eval().cuda()
        prediction_model.set_chunk_size(256)
    return mpnn_model, mpnn_config_dict, prediction_model, af2_setting
    
def get_fake_msa(file_path: str,fake_msa,output_dir):
    command = [
                "python", "/storage/caolongxingLab/fangminchao/work/LigandMPNN/run.py", "--model_type", "ligand_mpnn",
                "--seed", "111",
                "--pdb_path", f"{file_path}",
                "--out_folder", f"{output_dir}",
                "--ligand_mpnn_use_side_chain_context", "1",
                "--ligand_mpnn_use_atom_context", "1",
                "--parse_atoms_with_zero_occupancy", "1",
                "--number_of_batches", f"{fake_msa}"
            ]
    result = subprocess.run(
        " ".join(command),
        shell=True,
        text=True,
        capture_output=True,
        timeout=3600  
    )

    print("Command executed successfully!")
    print("Output:", result.stdout)
    fasta_path = os.path.join(output_dir, "seqs", os.path.basename(file_path.replace(".pdb",".fa"))) 
    sequences=read_fasta_sequences(fasta_path)
    print(sequences)
    fake_msa = ""
    msa = []
    for i, seq in enumerate(sequences, 1):
        msa.append(f">Seq{i}")
        msa.append(seq[0])
    return "\n".join(msa)

def get_random_seeds(num_seeds: int):
    import random
    random_numbers = [random.randint(0, 10000) for _ in range(num_seeds)]
    
    return random_numbers

def get_chain_sequence(file_path: str, chain_id: str) -> str:
    try:
        amino_acid_map = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        standard_amino_acids = set(amino_acid_map.keys())
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", file_path)
        
        for model in structure:
            if chain_id in model:
                chain = model[chain_id]
                sequence = ""
                for residue in chain:
                    if residue.get_resname() in standard_amino_acids:
                        sequence += amino_acid_map[residue.get_resname()]
                return sequence
        return None
    except Exception as e:
        print(f"error: {str(e)}")
        return None

from Bio.PDB import MMCIFParser

def calculate_average_b_factor(cif_file: str, chain_id: list = ["A"]) -> float:
    """
    Calculate the average B-factor of specified chains in a CIF file.

    Args:
        cif_file (str): Path to the CIF file.
        chain_id (list): List of chain IDs to include in the calculation. Default is ["A"].

    Returns:
        float: Average B-factor of the specified chains. Returns 0.0 if the chain does not exist or has no atoms.
    """
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file)
        
        b_factors = []
        for model in structure:
            for chain in model:
                if chain.id in chain_id:  # Process only specified chains
                    for residue in chain:
                        for atom in residue:
                            b_factors.append(atom.bfactor)
        
        return sum(b_factors) / len(b_factors) if b_factors else 0.0
    except Exception as e:
        print(f"Failed to calculate average B-factor: {str(e)}")
        return 0.0

def calculate_key_b_factor(cif_file: str, cutoff: float = 5.0) -> float:
    """
    Calculate the average B-factor of protein atoms that are within a certain distance 
    from non-protein (ligand/small molecule) atoms in a CIF file.

    Args:
        cif_file (str): Path to the CIF file.
        cutoff (float): Distance cutoff (in Å) to consider proximity to ligands. Default is 5.0 Å.

    Returns:
        float: Average B-factor of residues near small molecules. Returns 0.0 if no atoms meet the criteria.
    """
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file)
    
        ligand_atoms = []
        protein_atoms = []

        for model in structure:
            for chain in model:
                if chain.id == "A":  # Only consider chain A as protein
                    for residue in chain:
                        for atom in residue:
                            protein_atoms.append((residue.id, atom.serial_number, atom))
                else:
                    for residue in chain:
                        for atom in residue:
                            ligand_atoms.append((residue.id, atom.serial_number, atom))

        # Identify protein atoms within cutoff distance to ligand atoms
        nearby_atoms_within_cutoff = []
        for lig_res_id, lig_atom_id, lig_atom in ligand_atoms:
            for res_id, atom_id, atom in protein_atoms:
                distance = lig_atom - atom
                if distance <= cutoff:
                    nearby_atoms_within_cutoff.append((res_id, atom_id, atom))
        
        b_factors_near_ligand = [atom.bfactor for _, _, atom in nearby_atoms_within_cutoff]
        return sum(b_factors_near_ligand) / len(b_factors_near_ligand) if b_factors_near_ligand else 0.0
    except Exception as e:
        print(f"Failed to calculate B-factor near ligands: {str(e)}")
        return 0.0


def process_confidence_metrics(confidence_json_path: str, cif_path: str, copied_file : str,scaffold_path) -> Dict:

    try:
        with open(confidence_json_path, 'r') as f:
            confidence_json = json.load(f)
        
        ptm = confidence_json['chain_ptm'][0]
        chain_pair_iptm = confidence_json['chain_pair_iptm'][0][1:]
        iptm = sum(chain_pair_iptm) / len(chain_pair_iptm) if chain_pair_iptm else 0
        
        chain_pair_pae_min = confidence_json['chain_pair_pae_min'][0][1:]
        ipae = sum(chain_pair_pae_min) / len(chain_pair_pae_min) if chain_pair_pae_min else 0
        
        plddt = calculate_average_b_factor(cif_path,["A","B"])
        A_plddt = calculate_average_b_factor(cif_path,["A"])
        B_plddt = calculate_average_b_factor(cif_path,["B"])
        key_res_plddt= calculate_key_b_factor(cif_path, cutoff=5)
        rmsd, rmsd_lig,Lig_centre_distance= calculate_ca_rmsd(cif_path, copied_file)
        rmsd_between_origin, rmsd_lig_between_origin,Lig_centre_distance_between_origin = calculate_ca_rmsd(cif_path, scaffold_path)
        return {
            'AF3_PTM': ptm,
            'AF3_iPTM': iptm,
            'AF3_iPAE': ipae,
            'AF3_plddt': plddt,
            'AF3_A_plddt': A_plddt,
            'AF3_B_plddt': B_plddt,
            'AF3_key_res_plddt': key_res_plddt,
            'AF3_rmsd': rmsd,
            'AF3_lig_rmsd': rmsd_lig,
            'AF3_Lig_centre_distance': Lig_centre_distance,
            'AF3_Lig_centre_distance_between_origin': Lig_centre_distance_between_origin,
            'AF3_rmsd_between_origin': rmsd_between_origin,
            'AF3_lig_rmsd_between_origin': rmsd_lig_between_origin,
            'AF3_Status': 'Success'
        }
    except Exception as e:
        print(f"error: {str(e)}")
        return {
            'AF3_PTM': np.nan,
            'AF3_iPTM': np.nan,
            'AF3_iPAE': np.nan,
            'AF3_plddt': np.nan,
            'AF3_A_plddt': np.nan,
            'AF3_B_plddt': np.nan,
            'AF3_key_res_plddt': np.nan,
            'AF3_rmsd': np.nan,
            'AF3_lig_rmsd': np.nan,
            'AF3_Lig_centre_distance': np.nan,
            'AF3_Lig_centre_distance_between_origin': np.nan,
            'AF3_rmsd_between_origin': np.nan,
            'AF3_lig_rmsd_between_origin': np.nan,
            'AF3_Status': 'Failed'
        }
from Bio.PDB import MMCIFParser, PDBParser, Superimposer
from Bio.PDB.Atom import Atom
import numpy as np


import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, Superimposer
from Bio.PDB.Atom import Atom

def get_ca_atoms(structure):

    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:  
                    ca_atoms.append(residue["CA"])
    return ca_atoms


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from Bio.PDB import MMCIFParser, PDBParser, Superimposer, PDBIO
import numpy as np
import os

def get_ca_atoms(structure):
    ca_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atoms.append(residue['CA'])
    return ca_atoms

def get_ligand_atoms(structure):
    ligand_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() not in [
                    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 
                    'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 
                    'THR', 'VAL', 'TRP', 'TYR', 'HOH'
                ]:
                    for atom in residue:
                        if atom.element != 'H':
                            ligand_atoms.append(atom)
    return ligand_atoms



from Bio.PDB import PDBIO, Structure, Model, Chain, Residue

def save_ligand_structure(structure, output_file):
    """
    Extract ligand(s) from a protein structure and save them as a new PDB file.

    Non-protein residues (i.e., not standard amino acids or water) are treated as ligands,
    and only non-hydrogen atoms are retained.

    Args:
        structure (Bio.PDB.Structure.Structure): Original protein structure.
        output_file (str): Path to the output PDB file.
    """
    
    # Create a new PDBIO object to save the ligand structure
    io = PDBIO()
    
    # Create a new Structure object to store ligands
    new_structure = Structure.Structure('ligand_only')
    new_model = Model.Model(0)  # New model
    new_chain = Chain.Chain('A')  # New chain
    
    # List of standard amino acids and water to exclude
    standard_residues = [
        'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 
        'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 
        'THR', 'VAL', 'TRP', 'TYR', 'HOH'
    ]
    
    # Iterate over the original structure
    for model in structure:
        for chain in model:
            for residue in chain:
                # Identify non-protein residues (ligands)
                if residue.get_resname() not in standard_residues:
                    # Create a new Residue object to store non-hydrogen atoms
                    new_residue = Residue.Residue((' ', 1, ' '), 'LIG', '')
                    
                    # Copy non-hydrogen atoms to the new residue
                    for atom in residue:
                        if atom.element != 'H':
                            new_residue.add(atom.copy())

                    # Add the processed residue to the new chain
                    new_chain.add(new_residue)
    
    # Assemble the new structure
    new_model.add(new_chain)
    new_structure.add(new_model)
    
    # Save the ligand-only structure
    io.set_structure(new_structure)
    io.save(output_file)
    
    print(f"Ligand structure saved to {output_file}")



def read_pdb_coordinates(pdb_file):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('PDB', pdb_file)
    
    atom_coordinates = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_coordinates[atom.get_id()] = atom.get_coord()
    
    return atom_coordinates

def calculate_rmsd_from_coordinates(coords1, coords2, atom_map):

    diff = []
    atom1 =[]
    atom2 =[]
    for idx1, idx2 in atom_map:

        atom1_name = list(coords1.keys())[idx1]
        atom2_name = list(coords2.keys())[idx2]
        print(atom1_name,atom2_name)
        diff.append(coords1[atom1_name] - coords2[atom2_name])
        atom1.append(coords1[atom1_name])
        atom2.append(coords2[atom2_name])
    
    centre1 = np.mean(atom1, axis=0)
    centre2 = np.mean(atom2, axis=0)
    atom_centre_distance = np.linalg.norm(centre1 - centre2)
    diff = np.array(diff)
    squared_diff = np.square(diff)
    rmsd = np.sqrt(np.sum(squared_diff) / len(atom_map))
    return rmsd,atom_centre_distance


def align_molecules_from_pdb_files(smiles, pdb_file1, pdb_file2):
    # Create a reference molecule from the SMILES string
    ref_mol = Chem.MolFromSmiles(smiles)
    if ref_mol is None:
        raise ValueError("Cannot create reference molecule from SMILES")
    
    # Read molecules from PDB files
    mol1 = Chem.MolFromPDBFile(pdb_file1, removeHs=True)
    mol2 = Chem.MolFromPDBFile(pdb_file2, removeHs=True)
    
    if mol1 is None or mol2 is None:
        raise ValueError("Cannot create molecule from PDB file")

    # Get canonical SMILES to check structural consistency
    mol1_smiles = Chem.MolToSmiles(mol1, canonical=True)
    mol2_smiles = Chem.MolToSmiles(mol2, canonical=True)
    print(f"mol1 SMILES: {mol1_smiles}")
    print(f"mol2 SMILES: {mol2_smiles}")

    # Substructure matching to the reference molecule
    matches1 = mol1.GetSubstructMatch(ref_mol)
    matches2 = mol2.GetSubstructMatch(ref_mol)
    
    # Generate atom mapping between the two molecules
    atom_map = [(idx1, idx2) for idx1, idx2 in zip(matches1, matches2)]
    print(f"Atom map: {atom_map}")

    if not atom_map:
        raise ValueError("Atom mapping is empty!")

    # Read atomic coordinates from the PDB files
    coords1 = read_pdb_coordinates(pdb_file1)
    coords2 = read_pdb_coordinates(pdb_file2)

    # Calculate RMSD and per-atom distances
    rmsd, atom_distance = calculate_rmsd_from_coordinates(coords1, coords2, atom_map)
    return atom_distance, rmsd, atom_map



def calculate_ca_rmsd(cif_file, pdb_file):
    try:
        # Parse CIF and PDB files
        cif_parser = MMCIFParser(QUIET=True)
        cif_structure = cif_parser.get_structure("CIF", cif_file)
        
        pdb_parser = PDBParser(QUIET=True)
        pdb_structure = pdb_parser.get_structure("PDB", pdb_file)

        # Get Cα atoms
        cif_ca_atoms = get_ca_atoms(cif_structure)
        pdb_ca_atoms = get_ca_atoms(pdb_structure)

        if len(cif_ca_atoms) != len(pdb_ca_atoms):
            raise ValueError("Number of Cα atoms in CIF and PDB does not match")

        # Extract coordinates and align
        super_imposer = Superimposer()
        super_imposer.set_atoms(cif_ca_atoms, pdb_ca_atoms)  # Note the order of arguments
        super_imposer.apply(pdb_structure.get_atoms())  # Update PDB structure coordinates

        protein_rmsd = super_imposer.rms

        # Handle ligands
        cif_ligand_atoms = get_ligand_atoms(cif_structure)
        pdb_ligand_atoms = get_ligand_atoms(pdb_structure)

        if cif_ligand_atoms and pdb_ligand_atoms:
            if len(cif_ligand_atoms) != len(pdb_ligand_atoms):
                print(len(cif_ligand_atoms), len(pdb_ligand_atoms))
                print(cif_file, pdb_file)
                print("Number of ligand atoms does not match, skipping ligand RMSD calculation")
                return protein_rmsd, None, None

            # Save ligand structures to temporary PDB files
            cif_ligand_file = "cif_ligand.pdb"
            pdb_ligand_file = "pdb_ligand.pdb"
            
            save_ligand_structure(cif_structure, cif_ligand_file)
            save_ligand_structure(pdb_structure, pdb_ligand_file)

            # Create reference molecule from PDB ligand and align
            ref_mol = Chem.MolFromPDBFile(pdb_ligand_file, removeHs=True)
            if ref_mol is None:
                raise ValueError("Cannot create molecule from CIF ligand")
            ref_smiles = Chem.MolToSmiles(ref_mol, canonical=True)

            atom_centre_distance, ligand_rmsd, atom_map = align_molecules_from_pdb_files(
                ref_smiles, cif_ligand_file, pdb_ligand_file
            )
    finally:
        # Clean up temporary ligand files
        if os.path.exists(cif_ligand_file):
            os.remove(cif_ligand_file)
        if os.path.exists(pdb_ligand_file):
            os.remove(pdb_ligand_file)

    return protein_rmsd, ligand_rmsd, atom_centre_distance



from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


def template_process(cif_path, plddt_threshold=70, min_continuous_length=5):
    # Read the CIF file using MMCIF2Dict
    mmcif_dict = MMCIF2Dict(cif_path)
    
    # Extract B-factors (pLDDT), chain IDs, and residue IDs
    b_factors = mmcif_dict["_atom_site.B_iso_or_equiv"]  # pLDDT values
    chain_ids = mmcif_dict["_atom_site.auth_asym_id"]    # Chain IDs
    residue_ids = mmcif_dict["_atom_site.auth_seq_id"]   # Residue IDs
    
    # Convert pLDDT values to float
    b_factors = list(map(float, b_factors))
    
    # Create a dictionary storing pLDDT values by chain and residue
    chain_residue_plddt = {}
    for i in range(len(b_factors)):
        chain_id = chain_ids[i]
        residue_id = int(residue_ids[i]) - 1  # Convert residue IDs to zero-based
        plddt = b_factors[i]
        if chain_id == "A":  # Only process chain A
            if chain_id not in chain_residue_plddt:
                chain_residue_plddt[chain_id] = {}
            if residue_id not in chain_residue_plddt[chain_id]:
                chain_residue_plddt[chain_id][residue_id] = []
            
            chain_residue_plddt[chain_id][residue_id].append(plddt)
    
    # Compute average pLDDT for each residue
    chain_residue_avg_plddt = {}
    for chain_id, residue_dict in chain_residue_plddt.items():
        chain_residue_avg_plddt[chain_id] = {}
        for residue_id, plddt_list in residue_dict.items():
            avg_plddt = sum(plddt_list) / len(plddt_list)
            chain_residue_avg_plddt[chain_id][residue_id] = avg_plddt
    
    # Create a dictionary storing residues that pass the pLDDT threshold
    chain_to_residues = {}
    for chain_id, residue_dict in chain_residue_avg_plddt.items():
        for residue_id, avg_plddt in residue_dict.items():
            if avg_plddt > plddt_threshold:
                if chain_id not in chain_to_residues:
                    chain_to_residues[chain_id] = []
                chain_to_residues[chain_id].append(residue_id)
    
    # Filter residues to only include continuous segments of minimum length
    filtered_indices = []

    for chain_id, residues in chain_to_residues.items():
        residues.sort()
        current_sequence = []
        for residue_id in residues:
            if not current_sequence:
                current_sequence.append(residue_id)
            else:
                if residue_id == current_sequence[-1] + 1:
                    current_sequence.append(residue_id)
                else:
                    # Add sequence if it meets minimum length
                    if len(current_sequence) >= min_continuous_length:
                        filtered_indices.extend(current_sequence)
                    current_sequence = [residue_id]
        
        # Check the last sequence
        if len(current_sequence) >= min_continuous_length:
            filtered_indices.extend(current_sequence)

    # Prepare the result dictionary
    result = [{
        "mmcif": "",
        "queryIndices": filtered_indices,
        "templateIndices": filtered_indices
    }]
    
    return result


import pyrosetta

from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

from pyrosetta.rosetta.protocols.relax import FastRelax

from pyrosetta import MoveMap

class InterfaceAnalyzer:
    def __init__(self, params_file=None):
        self.params_file = params_file

        self._init_pyrosetta()
        self.analyzer = InterfaceAnalyzerMover()
        self._configure_analyzer()  

        self._configure_relax()    

    def _init_pyrosetta(self):
        """Initialize PyRosetta and load parameter files"""
        extra_options = "-mute all"
        if self.params_file:
            extra_options += f" -extra_res_fa {self.params_file}"
        pyrosetta.init(extra_options=extra_options)

    def _configure_analyzer(self):
        """Configure parser options"""
        self.scorefxn = pyrosetta.get_fa_scorefxn()
        self.analyzer.set_scorefunction(self.scorefxn)
        options = {
            'set_compute_packstat': True,
            'set_compute_interface_energy': True,
            'set_calc_dSASA': True,
            'set_calc_hbond_sasaE': True,
            'set_compute_interface_sc': True,
            'set_pack_separated': True,
            'set_use_centroid_dG': False

        }
        for opt, val in options.items():
            getattr(self.analyzer, opt)(val)

    def _configure_relax(self):
        """Configure the Relax protocol"""
        self.relax_protocol = FastRelax()
        self.relax_protocol.set_scorefxn(self.scorefxn)

        # Allow all backbone and sidechain movements (including ligands)
        mm = MoveMap()
        mm.set_bb(True)
        mm.set_chi(True)
        self.relax_protocol.set_movemap(mm)

        # Optional: set the number of Relax cycles (default is usually 5)
        # self.relax_protocol.set_script_file('MonomerRelax2019')


    def analyze_interface(self, pdb_path, protein_chain='A', ligand_chain='B'):
        try:
            self._init_pyrosetta()  

            pose = self._load_and_validate_pdb(pdb_path, protein_chain, ligand_chain)
            
            # Apply the Relax protocol to optimize the structure

            print("Relaxing...")
            self.relax_protocol.apply(pose)

            self.analyzer.set_interface(f"{protein_chain}_{ligand_chain}")
            self.analyzer.apply(pose)
            return self._compile_results()
        except Exception as e:
            print(f"error: {e}")
            return None

    def _load_and_validate_pdb(self, pdb_path, protein_chain, ligand_chain):
        """Load and validate a PDB structure"""
        pose = pyrosetta.pose_from_pdb(pdb_path)

        # Check if chains exist
        chains = {pose.pdb_info().chain(res) for res in range(1, pose.total_residue() + 1)}
        for chain in [protein_chain, ligand_chain]:
            if chain not in chains:
                raise ValueError(f"Chain {chain} does not exist in the PDB")

        # Check if ligand chain has residues
        lig_residues = [
            res for res in range(1, pose.total_residue() + 1) 
            if pose.pdb_info().chain(res) == ligand_chain
        ]
        if not lig_residues:
            raise ValueError(f"No ligand residues found in chain {ligand_chain}")

        print(f"Successfully loaded: {pdb_path}, total residues: {pose.total_residue()}")
        return pose


    def _compile_results(self):
        """Compile result data from the analyzer"""
        data = self.analyzer.get_all_data()
        return (
            self.analyzer.get_interface_dG(),
            data.dSASA,
            data.packstat,
            data.interface_hbonds,
            data.sc_value,
            data.dG_dSASA_ratio,
            list(data.interface_residues)
        )



def calculate_rosetta_metric(pdb_path,params_path):
    analyzer = InterfaceAnalyzer(params_file=params_path)
    
    # Perform analysis (assuming protein chain A and ligand chain B)

    interface_dG,dSASA,packstat,interface_hbonds,sc_value,energy_density,interface_residues = analyzer.analyze_interface(
        pdb_path=pdb_path,
        protein_chain='A',
        ligand_chain='B'
    )
    
    return interface_dG,dSASA,packstat,interface_hbonds

def atoms_templates_process(cif_path, plddt_threshold):
    # Read the CIF file using MMCIF2Dict

    mmcif_dict = MMCIF2Dict(cif_path)
    
    # Extract B-factor (pLDDT) and chain ID

    b_factors = mmcif_dict["_atom_site.B_iso_or_equiv"]  # plddt 

    chain_ids = mmcif_dict["_atom_site.auth_asym_id"]    # CHain ID

    
    # Convert pLDDT values to floats

    b_factors = list(map(float, b_factors))
    
    # Select B-chain atom indices that meet the criteria
    atom_sm_index =0
    b_chain_filtered_indices = []
    for i in range(len(b_factors)):
        chain_id = chain_ids[i]
        if chain_id == "B": 

            plddt = b_factors[i]
            if plddt > plddt_threshold:
                b_chain_filtered_indices.append(atom_sm_index)
            atom_sm_index +=1
    
    return b_chain_filtered_indices

def run_AF3_evaluation(output_dir,json_path):
    try:
        command = [
            "module", "load", "alphafold/3_a40-tmp", "&&",
            "singularity", "exec",
            "--nv",
            "--bind", f"{output_dir}:/root/output",
            "--bind", "/storage/caolongxingLab/fangminchao/",
            "--bind", "/storage/caolongxingLab/share/",
            "--bind", "/storage/caolongxingLab/fangminchao/tools/alphafold3/model:/root/models",
            "--bind", "/storage/caolongxingLab/fangminchao/database/AF3/public_databases:/root/public_databases",
            "/soft/bio/alphafold/3/alphafold3.sif",
            "python", "/storage/caolongxingLab/fangminchao/work/alphafold3_quality_evaluation/run_alphafold_avail.py",
            "--json_path="+json_path,
            "--model_dir=/root/models",
            "--db_dir=/root/public_databases",
            "--output_dir=/root/output",
            "--jax_compilation_cache_dir=/root/output/jax_compilation",
            "--norun_data_pipeline"
        ]
        
        result = subprocess.run(
            " ".join(command),
            shell=True,
            text=True,
            capture_output=True,
            timeout=3600  
        )
        result.check_returncode()  
        print("Command executed successfully!")
        print("Output:", result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        print("Command failed with a non-zero exit code!")
        print("Return code:", e.returncode)
        print("Error message:", e.stderr)
        print("Full output:", e.stdout)

    except subprocess.TimeoutExpired as e:
        print("Command timed out!")
        print("Timeout:", e.timeout)

    except Exception as e:
        print("An unexpected error occurred:", str(e))
    

def run_AF3_evaluation_with_ref_eval(output_dir,json_path,pkl_path,ref_eval):
    try:
        command = [
            "module", "load", "alphafold/3_a40-tmp", "&&",
            "singularity", "exec",
            "--nv",
            "--bind", f"{output_dir}:/root/output",
            "--bind", "/storage/caolongxingLab/fangminchao/",
            "--bind", "/storage/caolongxingLab/share/",
            "--bind", "/storage/caolongxingLab/fangminchao/tools/alphafold3/model:/root/models",
            "--bind", "/storage/caolongxingLab/fangminchao/database/AF3/public_databases:/root/public_databases",
            "/soft/bio/alphafold/3/alphafold3.sif",
            "python", "/storage/caolongxingLab/fangminchao/work/alphafold3_quality_evaluation/run_alphafold_avail.py",
            "--json_path="+json_path,
            "--ref_batch_eval",
            "--num_samples=5",
            "--ref_time_steps="+str(ref_eval),
            "--ref_pdb_path="+pkl_path,
            "--model_dir=/root/models",
            "--db_dir=/root/public_databases",
            "--output_dir=/root/output",
            "--jax_compilation_cache_dir=/root/output/jax_compilation",
            "--norun_data_pipeline"
        ]
        
        result = subprocess.run(
            " ".join(command),
            shell=True,
            text=True,
            capture_output=True,
            timeout=3600 
        )
        result.check_returncode()  
        print("Command executed successfully!")
        print("Output:", result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        print("Command failed with a non-zero exit code!")
        print("Return code:", e.returncode)
        print("Error message:", e.stderr)
        print("Full output:", e.stdout)

    except subprocess.TimeoutExpired as e:
        print("Command timed out!")
        print("Timeout:", e.timeout)

    except Exception as e:
        print("An unexpected error occurred:", str(e))
    
def run_Ligandmpnn_evaluation(scaffold_path,mpnn_config_dict,plddt_good_indicates,pocket_res,output_dir):
    pocket_plddt_good_res = common_elements(pocket_res ,plddt_good_indicates)
    pocket_plddt_good_res_set = set(pocket_plddt_good_res)
    pocket_res_to_design = [item for item in pocket_res if item not in pocket_plddt_good_res_set]
    non_pocket_to_fix =  list(set(plddt_good_indicates + pocket_res))
    pocket_res_to_design = " ".join([f"A{resi}" for  resi in pocket_res_to_design]) 
    non_pocket_to_fix = " ".join([f"A{resi}" for  resi in non_pocket_to_fix]) 
    print("plddt_good_indicates",plddt_good_indicates)
    print("pocket_res",pocket_res)
    print("pocket_res_to_design",pocket_res_to_design)
    print("non_pocket_to_fix",non_pocket_to_fix)
    sequences = []

    for i  in range(0,mpnn_config_dict):
        output_dir_ligmpnn =os.path.join(output_dir,f"ligandmpnn{i}")

        command = [
            "python", "/storage/caolongxingLab/fangminchao/work/LigandMPNN/run.py", "--model_type", "ligand_mpnn",
            "--seed", "111",
            "--pdb_path", f"{scaffold_path}",
            "--out_folder", f"{output_dir_ligmpnn}",
            "--redesigned_residues", f"'{pocket_res_to_design}'",
            "--pack_side_chains","1",
            "--pack_with_ligand_context","1",
            "--repack_everything", "1",
            "--parse_atoms_with_zero_occupancy", "1",
            "--ligand_mpnn_use_side_chain_context", "1",
            "--ligand_mpnn_use_atom_context", "1",
            "--number_of_batches", "1"
        ]
        result = subprocess.run(
        " ".join(command),
        shell=True,
        text=True,
        capture_output=True,
        timeout=3600  
        )
        output_dir_prompnn =os.path.join(output_dir,f"prompnn{i}")
        out_ligandmpnn_file = os.path.join(output_dir_ligmpnn,"packed", os.path.basename(scaffold_path.replace(".pdb","_packed_1_1.pdb")))
        command = [
            "python", "/storage/caolongxingLab/fangminchao/work/LigandMPNN/run.py", "--model_type", "protein_mpnn",
            "--seed", "111",
            "--pdb_path", f"{out_ligandmpnn_file}",
            "--out_folder", f"{output_dir_prompnn}",
            "--fixed_residues", f"'{non_pocket_to_fix}'",
            "--pack_side_chains","1",
            "--repack_everything", "1",
            "--parse_atoms_with_zero_occupancy", "1",
            "--ligand_mpnn_use_side_chain_context", "1",
            "--number_of_batches", "1"
        ]
        result = subprocess.run(
        " ".join(command),
        shell=True,
        text=True,
        capture_output=True,
        timeout=3600  
        )
        fasta_path = os.path.join(output_dir_prompnn,"seqs", os.path.basename(scaffold_path.replace(".pdb","_packed_1_1.fa")))
        sequences.append(read_fasta_sequences(fasta_path))
    print(sequences)
    return  sequences

import re

def read_fasta_sequences(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ""
        description = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"): 
                if sequence:
                    sequences.append((description, sequence)) 
                sequence = ""
                description = line  
            else:
                sequence += line 
        if sequence: 
            sequences.append((description, sequence)) 
    #  overall_confidence
    for i, (desc, seq) in enumerate(sequences):
        match = re.search(r'overall_confidence=([0-9.]+)', desc)
        if match:
            overall_confidence = float(match.group(1))
            sequences[i] = (seq, overall_confidence)  
    return sequences[1:]

    
def self_consistency_sm(scaffold_path,
                        mpnn_config_dict,
                        plddt_good_indicates,
                        output_dir,
                        template_path,
                        sm,  
                        ddg,
                        extra_res_params,
                        ref_eval):
    
    # Example usage:

    pocket_res = find_pocket_residues(pdbfile="/storage/caolongxingLab/share/fangminchao/af3_benchmark/sm/50_sm_recentre_6re_16msa_templates95/recycle_2/modified_AIN_120_86_recycle_2.pdb",
                                         cutoff_CA=8.0, cutoff_sc=6.0)
    print(f"pocket residues: {pocket_res}")
    sequences = run_Ligandmpnn_evaluation(scaffold_path,
                                                   mpnn_config_dict,
                                                   plddt_good_indicates,
                                                   pocket_res,
                                                   output_dir)
    seq_count =0 
    results_list = []

    for [seq_tuple] in sequences:
        # Read the template file
        seq,overall_confidence = seq_tuple
        with open(template_path, 'r') as f:
            template = json.load(f)
        
        result={}
        # Prepare input JSON for AF3
        input_json = template.copy()
        
        # Safely get the base name of the scaffold file, change extension to .fa
        scaffold_basename = os.path.splitext(os.path.basename(scaffold_path))[0]
        input_json['name'] = f"{scaffold_basename}_{seq_count}"

        # Update the sequence in the JSON
        input_json['sequences'][0]['protein']["sequence"] = seq
        input_json['sequences'][1]["ligand"]["smiles"] = sm
        # Add modelSeeds (assuming `get_random_seeds` function works)
        input_json["modelSeeds"] = get_random_seeds(1)
        # Prepare the path for the new JSON file
        json_filename = f"{scaffold_basename}_{seq_count}.json"
        json_path = os.path.join(output_dir, json_filename)
        # Write the new JSON file
        with open(json_path, 'w') as f:
            json.dump(input_json, f, indent=2)
        if ref_eval:
            print("normal AF3 batch diffusion plus ref guided diffusion")
            base_name = os.path.basename(scaffold_path)
            name_without_ext = os.path.splitext(base_name)[0]
            new_filename = f"{name_without_ext}_packed_1_1_packed_1_1.pdb"
            packed_scaffold_path = os.path.join(
                output_dir,
                f"prompnn{seq_count}",
                "packed",
                new_filename
            )
            ref_out_dir = os.path.join(output_dir, "packed")
            result_pkl, file_name, error = process_single_file(( packed_scaffold_path , ref_out_dir, ref_out_dir),None)
            # run AF3
            pkl_path = os.path.join(ref_out_dir, f'{os.path.splitext(packed_scaffold_path)[0]}.pkl')
            run_AF3_evaluation_with_ref_eval(output_dir, json_path, pkl_path,ref_eval)
        else:
            print("normal AF3 batch diffusion")
            run_AF3_evaluation(output_dir, json_path)
            
        
        tag =f"{scaffold_basename}_{seq_count}".lower()
        cif_path = os.path.join(output_dir,tag,f"{tag}_model.cif")
        confidence_json_path = os.path.join(output_dir,tag,f"{tag}_summary_confidences.json")
        with open(confidence_json_path, 'r') as f:
            confidence_json = json.load(f)
            
        result['mpnn_sequence'] = seq
        result["scaffold_path"] = cif_path
        result['mpnn_score'] = overall_confidence
        result['prediction_model'] = "AF3"
        result['A_plddt'] = calculate_average_b_factor(cif_path,["A"])
        result['B_plddt'] = calculate_average_b_factor(cif_path,["B"])
        result['AF2_plddt'] = calculate_average_b_factor(cif_path,["A","B"])
        result['key_res_plddt'] = calculate_key_b_factor(cif_path,5)
        result["A_ptm"] = confidence_json['chain_ptm'][0]
        result["B_ptm"] = confidence_json['chain_ptm'][1]
        chain_pair_iptm = confidence_json['chain_pair_iptm'][0][1:]
        result["iptm"] = sum(chain_pair_iptm) / len(chain_pair_iptm) if chain_pair_iptm else 0
        result["rmsd"] ,result["rmsd_lig"] ,result["atom_centre_distance"]  = calculate_ca_rmsd(cif_path, scaffold_path)
        pdb_path =cif_path.replace(".cif",".pdb")
        if ddg:
            convert_cif_to_pdb(cif_path,pdb_path)
            result["interface_dG"] ,result["dSASA"] ,result["packstat"],result["interface_hbonds"] = calculate_rosetta_metric(pdb_path,extra_res_params)
        seq_count += 1
        results_list.append(result)

    return results_list



def self_consistency(scaffold_path,output_path_tag, mpnn_model, mpnn_config_dict, prediction_model, af2_setting,plddt_good_indicates):
    
    mpnn_seqs, mpnn_scores = mpnn_design(
                config=mpnn_config_dict,
                protein_path=scaffold_path,
                model=mpnn_model,
                design_chains=['A'],
                fixed_residues=plddt_good_indicates,
            )
    #print(mpnn_seqs)
    import pandas as pd
    import re


    results_list = []
    for i in range(len(mpnn_seqs)):
        sequence = mpnn_seqs[i].split(':')[0] if ':' in mpnn_seqs[i] else mpnn_seqs[i]
        sequence = re.sub("[^A-Z]", "", sequence.upper())
        scaffold_prot = from_pdb_string(open(scaffold_path).read(), 'A' )
        evaluated_results, pred_prots = run_folding_and_evaluation(prediction_model, sequence, scaffold_prot, None, af2_setting, template_chains=None)
        #print(evaluated_results)
        for result in evaluated_results:
            result['sequence'] = sequence  
            result['index'] = i           
            results_list.append(result)
    for j, (result, pred_prot) in enumerate(zip(evaluated_results, pred_prots)):
        fold_path = os.path.join(os.path.dirname(scaffold_path), output_path_tag + f"_af2_{j}.pdb")
        with open(fold_path, 'w') as f:
            f.write(protein.to_pdb(pred_prot))
        result["alphafold_path"] = fold_path
        result['mpnn_sequence'] = mpnn_seqs[i]
   
    return results_list

import subprocess

def run_alphafold3(json_path: str, pkl_path: str, output_dir: str,ref_time_steps: str,num_samples: str,) -> bool:
    """AlphaFold3"""
    try:
        command = [
            "module", "load", "alphafold/3_a40-tmp", "&&",
            "singularity", "exec",
            "--nv",
            "--bind", f"{output_dir}:/root/output",
            "--bind", "/storage/caolongxingLab/fangminchao/",
            "--bind", "/storage/caolongxingLab/share/",
            "--bind", "/storage/caolongxingLab/fangminchao/tools/alphafold3/model:/root/models",
            "--bind", "/storage/caolongxingLab/fangminchao/database/AF3/public_databases:/root/public_databases",
            "/soft/bio/alphafold/3/alphafold3.sif",
            "python", "/storage/caolongxingLab/fangminchao/work/alphafold3_quality_evaluation/run_alphafold_avail.py",
            "--json_path="+json_path,
            "--ref_pdb_path="+pkl_path,
            "--ref_time_steps=" +ref_time_steps,
            "--num_samples=" +num_samples,
            "--model_dir=/root/models",
            "--db_dir=/root/public_databases",
            "--output_dir=/root/output",
            "--jax_compilation_cache_dir=/root/output/jax_compilation",
            "--norun_data_pipeline"
        ]
        
    #    result = subprocess.run(
    #        " ".join(command),
    #        shell=True,
    #        text=True,
    #        capture_output=True,
    #        timeout=3600  
    #    )
    #    return result.returncode == 0
    #except subprocess.TimeoutExpired:
    #    print("out of time")
    #    return False
    #except Exception as e:
    #    print(f"AF3 error: {str(e)}")
    #    return False
        # Run the command
        result = subprocess.run(
            " ".join(command),
            shell=True,
            text=True,
            capture_output=True,
            timeout=3600  
        )
        # Check if the process failed
        result.check_returncode()  
        print("Command executed successfully!")
        print("Output:", result.stdout)
        return True

    except subprocess.CalledProcessError as e:
        print("Command failed with a non-zero exit code!")
        print("Return code:", e.returncode)
        print("Error message:", e.stderr)
        print("Full output:", e.stdout)

    except subprocess.TimeoutExpired as e:
        print("Command timed out!")
        print("Timeout:", e.timeout)

    except Exception as e:
        print("An unexpected error occurred:", str(e))

import torch
import gc

def clear_gpu_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
import pathlib
from typing import Optional

def _read_file(path: pathlib.Path, json_path: Optional[pathlib.Path]) -> str:
  """Reads a maybe compressed (gzip, xz, zstd) file from the given path.

  Args:
    path: The path to the file to read. This can be either absolute path, or a
      path relative to the JSON file path.
    json_path: The path to the JSON file. If None, the path must be absolute.

  Returns:
    The contents of the file.
  """
  if not path.is_absolute():
    if json_path is None:
      raise ValueError('json_path must be specified if path is not absolute.')
    path = (json_path.parent / path).resolve()

  with open(path, 'rb') as f:
    first_six_bytes = f.read(6)
    f.seek(0)

    # Detect the compression type using the magic number in the header.
    if first_six_bytes[:2] == b'\x1f\x8b':
        return None
    else:
      return f.read().decode('utf-8')

import re


def process_single_pdb(pdb_file: str, 
                      cycle: int,
                      input_dir: str,
                      output_dir: str,
                      template_path: str,
                      mpnn_model,
                      mpnn_config_dict,
                      prediction_model,
                      af2_setting,
                      ref_time_steps,
                      num_samples,
                      num_seeds,
                      template_plddt_threshold,
                      sm_plddt_threshold,
                      reconstruct,
                      fake_msa,
                      ref_eval,
                      sm: str = "",  
                      ddg : bool = False,
                      extra_res_params: str = "",
                      run_af3: bool = True) -> Dict:
    """single cycle"""

    metrics = {
        'PDB': pdb_file,
        'Cycle': cycle + 1,
        'A_plddt': np.nan,
        'B_plddt': np.nan,
        'key_res_plddt': np.nan,
        'A_ptm': np.nan,
        'B_ptm': np.nan,
        'iptm': np.nan,
        'AF2_RMSD': np.nan,
        'AF2_LIG_RMSD': np.nan,
        'AF2_plddt': np.nan,
        'AF3_PTM': np.nan,
        'AF3_iPTM': np.nan,
        'AF3_iPAE': np.nan,
        'AF3_plddt': np.nan,
        'AF3_A_plddt': np.nan,
        'AF3_B_plddt': np.nan,
        'AF3_key_res_plddt': np.nan,
        'AF3_lig_rmsd': np.nan,
        'AF3_rmsd': np.nan,
        "interface_dG" : np.nan,
        "dSASA": np.nan,
        "packstat": np.nan,
        "interface_hbonds": np.nan,
        "atom_centre_distance" : np.nan,
        'AF3_rmsd_between_origin': np.nan,
        'AF3_lig_rmsd_between_origin': np.nan,
        'AF3_Lig_centre_distance': np.nan,
        'AF3_Lig_centre_distance_between_origin': np.nan,
        'AF2_Results' : [],
        'AF3_Status': 'Not Run'
    }
    
    try:
        # mkdir dir
        target_dir = os.path.join(output_dir, f"recycle_{cycle+1}")
        os.makedirs(target_dir, exist_ok=True)
        
        # replicate
        source_file = os.path.join(input_dir, pdb_file) if cycle == 0 else input_dir
        copied_file = os.path.join(target_dir, pdb_file.replace(".pdb", f"_recycle_{cycle+1}.pdb"))

        shutil.copy(source_file, copied_file)
        plddt_good_indicates = []
        plddt_atoms_indicates = []
        if template_plddt_threshold > 0 and cycle >=1:
            tag_pre = f"{pdb_file}_recycle_{cycle}".lower()
            previous_dir = os.path.join(output_dir, f"recycle_{cycle}")
            cif_pre_path= os.path.join(previous_dir, tag_pre.replace(".pdb", ""), 
                                  tag_pre.replace(".pdb", "") + "_model.cif")
            template_to_json= template_process(cif_pre_path,template_plddt_threshold)
            plddt_good_indicates = template_to_json[0]["templateIndices"]
            print(plddt_good_indicates)
            if sm_plddt_threshold > 0:
                plddt_atoms_indicates = atoms_templates_process(cif_pre_path, sm_plddt_threshold)
                print(plddt_atoms_indicates)
        
        if sm:
            AF3_sm_dir=copied_file.replace(".pdb","AF3_eval")
            os.makedirs(AF3_sm_dir, exist_ok=True)
            results_af2=self_consistency_sm(
                copied_file,
                mpnn_config_dict,
                plddt_good_indicates,
                AF3_sm_dir,
                template_path,
                sm,
                ddg,
                extra_res_params,
                ref_eval
            )

            min_rmsd = float('inf')
            A_plddt = 0
            B_plddt = 0
            all_plddt = 0
            iptm = 0
            min_rmsd_lig = float('inf')
            A_ptm = 0
            B_ptm =  0
            key_res_plddt = 0
            atom_centre_distance = 0
            scaffold_path = None
            interface_dG=0
            dSASA=0
            packstat=0
            interface_hbonds=0
            for entry in results_af2:
                if entry['rmsd_lig'] < min_rmsd_lig:
                    min_rmsd = entry['rmsd']
                    min_rmsd_lig = entry['rmsd_lig']
                    A_plddt = entry['A_plddt']
                    B_plddt = entry['B_plddt']
                    all_plddt = entry['AF2_plddt']
                    iptm = entry['iptm']
                    A_ptm = entry['A_ptm']
                    B_ptm = entry['B_ptm']
                    key_res_plddt = entry['key_res_plddt'] 
                    scaffold_path =entry["scaffold_path"]
                    atom_centre_distance =entry["atom_centre_distance"]
                    interface_dG=entry["interface_dG"] 
                    dSASA=entry["dSASA"] 
                    packstat=entry["packstat"]
                    interface_hbonds=entry["interface_hbonds"]

            metrics["atom_centre_distance"] = atom_centre_distance
            metrics['AF2_LIG_RMSD'] = min_rmsd_lig
            metrics['AF2_RMSD'] = min_rmsd
            metrics['A_plddt'] = A_plddt
            metrics['B_plddt'] = B_plddt
            metrics['AF2_plddt'] = all_plddt
            metrics['iptm'] = iptm
            metrics['A_ptm'] = A_ptm
            metrics['B_ptm'] = B_ptm
            metrics["scaffold_path"] = scaffold_path
            metrics["key_res_plddt"] = key_res_plddt
            metrics["interface_dG"] = interface_dG
            metrics["dSASA"] = dSASA
            metrics["packstat"] = packstat
            metrics["interface_hbonds"] = interface_hbonds
            for entry in results_af2:
                metrics['AF2_Results'].append({
                    'sequence': entry['mpnn_sequence'],
                    'rmsd': entry['rmsd'],
                    'LIG_rmsd': entry['rmsd_lig'],
                    'A_plddt': entry['A_plddt'],
                    'B_plddt': entry['B_plddt'],
                    'A_ptm': entry['A_ptm'],
                    'B_ptm': entry['B_ptm'],
                    "key_res_plddt": entry['key_res_plddt'],
                    "iptm": entry['iptm'],
                    "scaffold_path": entry['scaffold_path'],
                    "atom_centre_distance" : entry["atom_centre_distance"],
                    "interface_dG":entry["interface_dG"] ,
                    "dSASA" : entry["dSASA"] ,
                    "packstat" : entry["packstat"],
                    "interface_hbonds" :entry["interface_hbonds"]
                })
        else:
            # Run AF2 consistency check
            results_af2 = self_consistency(
                copied_file, 
                pdb_file.replace(".pdb", f"_recycle_{cycle+1}"),
                mpnn_model, 
                mpnn_config_dict, 
                prediction_model, 
                af2_setting,
                 [("A", resi) for resi in plddt_good_indicates]
            )


            min_rmsd = float('inf')
            corresponding_plddt = None
            sequence = None
            for entry in results_af2:
                if entry['rmsd'] < min_rmsd:
                    min_rmsd = entry['rmsd']
                    corresponding_plddt = entry['plddt']
                    sequence = entry['sequence']

            metrics['AF2_RMSD'] = min_rmsd
            metrics['AF2_plddt'] = corresponding_plddt

            for entry in results_af2:
                metrics['AF2_Results'].append({
                    'sequence': entry['sequence'],
                    'rmsd': entry['rmsd'],
                    'plddt': entry['plddt']
                })
            
        if not run_af3:
            return metrics, copied_file

        with open(template_path, 'r') as f:
            template = json.load(f)

        tag = f"{pdb_file}_recycle_{cycle+1}"
        json_path = os.path.join(target_dir, pdb_file.replace(".pdb", ".json"))
        if reconstruct == "af2":
            alphafold_path_entries = [entry['alphafold_path'] for entry in results_af2 if 'alphafold_path' in entry][0]
            copied_file = alphafold_path_entries
        elif reconstruct == "pose_seq_scaffold":
            sequence_string = [entry['mpnn_sequence'] for entry in results_af2][0]
            print(f"Processed sequence: {sequence_string}")
            base_name = os.path.basename(copied_file)
            name_without_ext = os.path.splitext(base_name)[0]
            new_filename = f"{name_without_ext}_packed_1_1_packed_1_1.pdb"
            packed_scaffold_path = os.path.join(
                AF3_sm_dir,
                f"prompnn0",
                "packed",
                new_filename
            )
            copied_file = packed_scaffold_path

        else:
            copied_file = copied_file
        input_json = template.copy()
        input_json['name'] = tag.replace(".pdb", "")
        input_json['sequences'][0]['protein']["sequence"] = get_chain_sequence(copied_file, 'A')
        if sm:
            input_json['sequences'][1]['ligand']["smiles"] = sm
        input_json["modelSeeds"] = get_random_seeds(num_seeds)
        if fake_msa and cycle >=1:
            tag_pre = f"{pdb_file}_recycle_{cycle}".lower()
            previous_dir = os.path.join(output_dir, f"recycle_{cycle}")
            seeds_path = os.path.join(previous_dir, tag_pre.replace(".pdb", ""))
            subfolders = [f for f in os.listdir(seeds_path) 
                 if os.path.isdir(os.path.join(seeds_path, f))]
            msa=""
            for subfolder in subfolders:
                subfolder_out = os.path.join(seeds_path,subfolder)
                cif_file = os.path.join(seeds_path, subfolder,subfolder+"_model.cif")
                convert_cif_to_pdb(cif_file,pdb_file)
                msa=msa+get_fake_msa(pdb_file,fake_msa,subfolder_out)
            input_json['sequences'][0]['protein']["pairedMsa"] = msa
        if template_plddt_threshold > 0 and cycle >=1:
            template_to_json[0]["mmcif"] = _read_file(pathlib.Path(cif_pre_path), pathlib.Path(json_path))
            input_json['sequences'][0]['protein']['templates'] = template_to_json
        with open(json_path, 'w') as f:
            json.dump(input_json, f, indent=2)
        
        # pkl_process

        result, file_name, error = process_single_file(( copied_file, target_dir, target_dir),plddt_atoms_indicates)
        
        pkl_path = os.path.join(target_dir, f'{os.path.splitext(copied_file)[0]}.pkl')
        clear_gpu_memory()
        success = run_alphafold3(json_path, pkl_path, target_dir, ref_time_steps=str(ref_time_steps), num_samples=str(num_samples))
        
        if success:
            tag =f"{tag}".lower()
            cif_path = os.path.join(target_dir, tag.replace(".pdb", ""), 
                                  tag.replace(".pdb", "") + "_model.cif")
            confidence_path = os.path.join(target_dir, tag.replace(".pdb", ""), 
                                         tag.replace(".pdb", "") + "_summary_confidences.json")
            
            af3_metrics = process_confidence_metrics(confidence_path,
                                                     cif_path, copied_file,
                                                     os.path.join(output_dir, "recycle_1" ,pdb_file.replace(".pdb", "_recycle_1.pdb")))
            metrics.update(af3_metrics)
            
            pdb_output = cif_path.replace(".cif", ".pdb")
            if convert_cif_to_pdb(cif_path, pdb_output):
                return metrics, pdb_output
            else:
                print(f"convert filed")
                return metrics, copied_file
        else:
            print(f"AF3 wrong")
            metrics['AF3_Status'] = 'Failed'
            return metrics, copied_file
            
    except Exception as e:
        print(f"error: {str(e)}")
        return metrics, copied_file


import pandas as pd
import os
from filelock import FileLock

def main():
    args = parse_arguments()
    
    if not os.path.exists(args.template_path):
        raise FileNotFoundError(f"Template file {args.template_path} not found!")
    extra_options = "-mute all"
    if args.extra_res_params:
        extra_options += f" -extra_res_fa {args.extra_res_params}"
    pyrosetta.init(extra_options=extra_options)
    if not args.sm :
        mpnn_model, mpnn_config_dict, prediction_model, af2_setting = self_consistency_init(args.alphafold,args.num_seqs)
    
    with open(args.pdb_list, 'r') as f:
        pdb_files = [line.strip() for line in f if line.strip()]
    
    csv_path = os.path.join(args.output_dir, 'processing_results.csv')
    lock_path = f"{csv_path}.lock" 
    lock = FileLock(lock_path)
    
    all_results = []
    for pdb_file in pdb_files:
        print(f"\nProcessing {pdb_file}...")
        current_input = args.input_dir
        for cycle in range(args.num_recycles):
            print(f"  Starting cycle {cycle+1}")
            try:
                is_last_cycle = (cycle == args.num_recycles - 1)
                if cycle > 0 :
                    if metrics['AF3_plddt'] > args.early_stop_threshold and metrics['AF3_rmsd'] < 1:
                        is_last_cycle =True
                if args.sm :
                    metrics, next_input = process_single_pdb(
                        pdb_file,
                        cycle,
                        current_input,
                        args.output_dir,
                        args.template_path,
                        None,
                        args.num_seqs,
                        "AF3",
                        None,
                        args.ref_time_steps,
                        args.num_samples,
                        args.num_seeds,
                        args.template_plddt_threshold,
                        args.sm_plddt_threshold,
                        args.reconstruct,
                        args.fake_msa,
                        args.ref_eval,
                        args.sm,
                        args.ddg,
                        args.extra_res_params,
                        run_af3=not is_last_cycle  
                    )
                else:
                    metrics, next_input = process_single_pdb(
                        pdb_file,
                        cycle,
                        current_input,
                        args.output_dir,
                        args.template_path,
                        mpnn_model,
                        mpnn_config_dict,
                        prediction_model,
                        af2_setting,
                        args.ref_time_steps,
                        args.num_samples,
                        args.num_seeds,
                        args.template_plddt_threshold,
                        args.reconstruct,
                        args.fake_msa,
                        run_af3=not is_last_cycle 
                    )
                print(next_input)
                all_results.append(metrics)
                current_input = next_input 
                


                
                print(f"  Cycle {cycle+1} completed:")
                print(f"    AF2 RMSD: {metrics['AF2_RMSD']:.3f}")
                print(f"    AF2 A plddt: {metrics['A_plddt']:.3f}")
                print(f"    AF2 B plddt: {metrics['B_plddt']:.3f}")
                print(f"    AF2 all plddt: {metrics['AF2_plddt']:.3f}")
                print(f"    AF2 iptm: {metrics['iptm']:.3f}")
                print(f"    AF3 Status: {metrics['AF3_Status']}")
                if metrics['AF3_Status'] == 'Success':
                    print(f"    AF3 PTM: {metrics['AF3_PTM']:.3f}")
                    print(f"    AF3 plddt: {metrics['AF3_plddt']:.3f}")
                    print(f"    AF3_rmsd: {metrics['AF3_rmsd']:.3f}")
                    
                        
                with lock:
                    file_exists = os.path.exists(csv_path)
                    pd.DataFrame([metrics]).to_csv(csv_path, mode='a', header=not file_exists, index=False)
                if is_last_cycle:
                   break 
            except Exception as e:
                print(f"  Error in cycle {cycle+1}: {str(e)}")
                continue
    
    print(f"Processing completed. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
