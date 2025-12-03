from Bio.PDB import MMCIFParser, PDBParser, PDBIO, NeighborSearch
import numpy as np
import string
import json
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import re
import pathlib
from typing import Optional
import torch
import gc
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from Bio.PDB import  Superimposer
import numpy as np
import os
from Bio.PDB import  MMCIFIO
from Bio.PDB import PDBParser, NeighborSearch
from scipy.stats import pearsonr
from Bio.PDB.Polypeptide import aa3  

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB")


def generate_cross_chain_symmetry(protein_info, symmetry_chains):
    """Generate a cross-chain symmetric residue identifier string

    Args:
        protein_info (dict): Protein structure dictionary {chain_id: [residue objects]}
        symmetry_chains (str): Symmetric chain identifiers, e.g., "A,B,C"

    Returns:
        str: Formatted like "A1,B1,C1|A2,B2,C2"

    Example:
        Given three chains each with 2 residues → returns "A1,B1,C1|A2,B2,C2"
    """
    # Parse the symmetric chains


    chain_ids = [c.strip() for c in symmetry_chains.split(',')]
    
    # Collect residues from all chains and sort them

    chain_residues = {}
    for cid in chain_ids:
        if residues := protein_info.get(cid):
            # Sort by residue number

            chain_residues[cid] = sorted(residues, key=lambda x: x.id[1])
    
    # Verify that all chains have the same length


    res_counts = [len(r) for r in chain_residues.values()]
    if not res_counts:
        return ""
    if len(set(res_counts)) > 1:
        raise ValueError(f"Number of residues in symmetric chains is inconsistent: {res_counts}")
    

    num_groups = res_counts[0]
    groups = []
    for idx in range(num_groups):
        group = []
        for cid in chain_ids:
            if cid in chain_residues and idx < len(chain_residues[cid]):
                # Use position index + 1 as the residue number

                group.append(f"{cid}{idx + 1}") 
        groups.append(",".join(group))
    
    return "|".join(groups)

def parse_symmetry_residues(symmetry_residues_str):
        # Parse the string into a list of residue groups

        groups = symmetry_residues_str.split('|')
        parsed_groups = [group.split(',') for group in groups]
        return parsed_groups

def calculate_weights(residue_groups):
    # Calculate the symmetry weight for each group
    weights = [1.0 / len(group) for group in residue_groups]
    return weights
    



def convert_cif_to_pdb(cif_file, pdb_file):
    """
    convert CIF to PDB, modify resnameLIG_B to LIG
    """

    amino_acids = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", 
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", 
        "TYR", "VAL", "ASX", "GLX", "PYR", "SEC", "SEL", "XLE", "XAA"
    ]
    try:

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("complex", cif_file)
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname()

                    if resname not in amino_acids:  
                        residue.resname = "LIG"  # modify resname for further process 
        
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_file)

        print(f"convert successfully, save as {pdb_file}")
        return True
    except Exception as e:
        print(f"convert failed, {str(e)}")
        return False


def template_process(cif_path, plddt_threshold=70,min_continuous_length=5):
    # Read the CIF file using MMCIF2Dict

    mmcif_dict = MMCIF2Dict(cif_path)

    # Extract B-factor (pLDDT), chain ID, and residue ID


    b_factors = mmcif_dict["_atom_site.B_iso_or_equiv"]  # pLDDT 

    chain_ids = mmcif_dict["_atom_site.auth_asym_id"]    # Chain ID

    residue_ids = mmcif_dict["_atom_site.auth_seq_id"]   # residue ID

    
    # Convert pLDDT values to floats
    b_factors = list(map(float, b_factors))

    # Create a dictionary to store pLDDT values by chain and residue


    chain_residue_plddt = {}
    for i in range(len(b_factors)):
        chain_id = chain_ids[i]
        residue_id = int(residue_ids[i]) - 1  

        plddt = b_factors[i]
        
        if chain_id not in chain_residue_plddt:
            chain_residue_plddt[chain_id] = {}
        if residue_id not in chain_residue_plddt[chain_id]:
            chain_residue_plddt[chain_id][residue_id] = []
        
        chain_residue_plddt[chain_id][residue_id].append(plddt)
    
    # Calculate the average pLDDT for each residue

    chain_residue_avg_plddt = {}
    for chain_id, residue_dict in chain_residue_plddt.items():
        chain_residue_avg_plddt[chain_id] = {}
        for residue_id, plddt_list in residue_dict.items():
            avg_plddt = sum(plddt_list) / len(plddt_list)
            chain_residue_avg_plddt[chain_id][residue_id] = avg_plddt

    # Analyze each chain and generate results
    results = []
    for chain_id, residue_dict in chain_residue_avg_plddt.items():
        # Create a list to store residues that meet the criteria
        chain_to_residues = []
        for residue_id, avg_plddt in residue_dict.items():
            if avg_plddt > plddt_threshold:
                chain_to_residues.append(residue_id)

        # Filter for consecutive residues

        filtered_indices = []
        chain_to_residues.sort()
        current_sequence = []
        for residue_id in chain_to_residues:
            if not current_sequence:
                current_sequence.append(residue_id)
            else:
                if residue_id == current_sequence[-1] + 1:
                    current_sequence.append(residue_id)
                else:
                    if len(current_sequence) >= min_continuous_length:
                        filtered_indices.extend(current_sequence)
                    current_sequence = [residue_id]
        
        if len(current_sequence) >= min_continuous_length:
            filtered_indices.extend(current_sequence)


        results.append({
            "mmcif": cif_path,
            "chain": chain_id,
            "queryIndices": filtered_indices,
            "templateIndices": filtered_indices
        })
    
    return results

def atoms_templates_process(cif_path, plddt_threshold):
    # Read the CIF file using MMCIF2Dict
    mmcif_dict = MMCIF2Dict(cif_path)

    # Extract B-factor (pLDDT), chain ID, and atom name


    b_factors = mmcif_dict["_atom_site.B_iso_or_equiv"]  # plddt value

    chain_ids = mmcif_dict["_atom_site.auth_asym_id"]    # Chain ID

    
    # # Attempt to retrieve atom names


    atom_names = mmcif_dict.get('_atom_site.auth_atom_id')
    if atom_names is None:
        atom_names = mmcif_dict.get('_atom_site.label_atom_id')  # # Try using alternative possible key names


    if atom_names is None:
        raise KeyError("Atom ID not found in the MMCIF file.")
    
    # Convert pLDDT values to floats
    b_factors = list(map(float, b_factors))

    # Initialize the maximum pLDDT value along with its corresponding atom name and index


    max_plddt = -1

    max_plddt_atom_name = None

    max_plddt_index = None

    # Select B-chain atom indices that meet the criteria and find the maximum pLDDT value

    atom_sm_index = 0

    for i in range(len(b_factors)):
        chain_id = chain_ids[i]
        if chain_id == "B":  # Process only chain B

            plddt = b_factors[i]
            if plddt > plddt_threshold:
                if plddt > max_plddt:
                    max_plddt = plddt

                    max_plddt_atom_name = atom_names[i]
                    max_plddt_index = atom_sm_index

            atom_sm_index += 1

    return max_plddt_atom_name, max_plddt_index


import json

def count_chain_based_on_json(data):
    protein_chains = 0

    ligand_chains = 0

    dna_chains = 0

    rna_chains = 0

    chain_types = []

    protein_found = False

    ligand_found = False

    dna_found = False

    rna_found = False

    with open(data, 'r') as f:
        data = json.load(f)
    sequences = data.get("sequences", [])
    for index, seq in enumerate(sequences):
        if "protein" in seq:
            if ligand_found or dna_found or rna_found:
                raise ValueError("Protein appears after ligand, DNA, or RNA in the JSON data.")
            protein_found = True
            protein_chains += len(seq["protein"].get("id", []))
            chain_types.extend(["protein"] * len(seq["protein"].get("id", [])))
        elif "ligand" in seq:
            ligand_found = True
            ligand_chains += len(seq["ligand"].get("id", []))
            chain_types.extend(["ligand"] * len(seq["ligand"].get("id", [])))
        elif "dna" in seq:
            dna_found = True
            dna_chains += len(seq["dna"].get("id", []))
            chain_types.extend(["dna"] * len(seq["dna"].get("id", [])))
        elif "rna" in seq:
            rna_found = True
            rna_chains += len(seq["rna"].get("id", []))
            chain_types.extend(["rna"] * len(seq["rna"].get("id", [])))
        else:
            raise ValueError(f"Unexpected entry in sequences at index {index}: neither protein, ligand, DNA, nor RNA.")

    return protein_chains, ligand_chains, dna_chains, rna_chains, chain_types

def count_chain_based_on_json_protenix(data):
    protein_chains = 0

    ligand_chains = 0

    dna_chains = 0

    rna_chains = 0

    chain_types = []

    protein_found = False

    ligand_found = False

    dna_found = False

    rna_found = False

    with open(data, 'r') as f:
        data = json.load(f)
    sequences = data[0]["sequences"]
    print(sequences)
    for index, seq in enumerate(sequences):
        print(seq)
        if "proteinChain" in seq:
            if ligand_found or dna_found or rna_found:
                raise ValueError("Protein appears after ligand, DNA, or RNA in the JSON data.")
            protein_found = True
            protein_chains += seq['proteinChain']["count"]
            chain_types.extend(["protein"] * seq['proteinChain']["count"])
        elif 'ligand' in seq:
            ligand_found = True
            ligand_chains += seq['ligand']["count"]
            chain_types.extend(["ligand"] * seq['ligand']["count"])
        elif "dnaSequence" in seq:
            dna_found = True
            dna_chains += seq['dnaSequence']["count"]
            chain_types.extend(["dna"] * seq['dnaSequence']["count"])
        elif "rnaSequence" in seq:
            rna_found = True
            rna_chains += seq['rnaSequence']["count"]
            chain_types.extend(["rna"] * seq['rnaSequence']["count"])
        else:
            raise ValueError(f"Unexpected entry in sequences at index {index}: neither protein, ligand, DNA, nor RNA.")

    return protein_chains, ligand_chains, dna_chains, rna_chains, chain_types

class interact_fix_analyze():
    def __init__(self, params_file=None,
                 ddg_chains_1=('A'),
                ddg_chains_2=('B')):
        self.params_file = params_file
        self.protein_chains = ddg_chains_1
        self.ligand_chains = ddg_chains_2
        self.sfxn = self.initialize_pyrosetta()
        print(f"rosetta energy {self.protein_chains} {self.ligand_chains}")

    def initialize_pyrosetta(self):
        """Initialize PyRosetta and load the scoring function."""

        extra_options = "-mute all"
        if self.params_file:
            extra_options += f" -extra_res_fa {self.params_file}"
        pyrosetta.init(extra_options=extra_options)
        sfxn = pyrosetta.get_score_function()
        opts = sfxn.energy_method_options()
        hb_opts = opts.hbond_options()
        hb_opts.decompose_bb_hb_into_pair_energies(True)
        opts.hbond_options(hb_opts)
        sfxn.set_energy_method_options(opts)
        return sfxn

    def calculate_pairwise_energies(self, pdb_file, energy_threshold=-3):
        """Calculate pairwise interaction energies between residues in a given PDB file, including chain information.

        Parameters:
            pdb_file (str): Path to the PDB file.
            energy_threshold (float): Energy threshold for filtering interactions, default is -1.

        Returns:
            list: A list of residue pairs with significant interactions and their energies.
        """

        pose = pyrosetta.pose_from_pdb(pdb_file)
        total_energy = self.sfxn(pose)

        energy_graph = pose.energies().energy_graph()
        significant_interactions = []
        all_interactions = []

        for i in range(1, pose.size() + 1):
            for j in range(i + 1, pose.size() + 1):
                edge = energy_graph.find_edge(i, j)
                if edge is not None:
                    e = edge.dot(self.sfxn.weights())
                    if e < energy_threshold:
                        # Get chain information

                        chain_i = pose.pdb_info().chain(i)
                        chain_j = pose.pdb_info().chain(j)
                        # Get the true PDB residue number

                        pdb_index_i = pose.pdb_info().number(i)
                        pdb_index_j = pose.pdb_info().number(j)

                        all_interactions.append((pdb_index_i, chain_i, pdb_index_j, chain_j, e))

                        if (chain_i in self.protein_chains and chain_j in self.ligand_chains) or (chain_j in self.protein_chains and chain_i in self.ligand_chains):
                            significant_interactions.append((pdb_index_i, chain_i, pdb_index_j, chain_j, e))

        return total_energy, significant_interactions, all_interactions

def get_chain_sequence(file_path: str, chain_id: str) -> str:
    """Retrieve the amino acid sequence of a specified chain from a PDB file"""
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
    
def read_fasta_sequences(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ""
        description = ""
        for line in file:
            line = line.strip()
            if line.startswith(">"):  # When encountering a description line
                # Save the previous sequence if it exists
                if sequence:
                    sequences.append((description, sequence))  # Store description and sequence
                sequence = ""
                description = line  # Update current description
            else:
                sequence += line  # Append sequence data
        if sequence:  # Add the last sequence manually
            sequences.append((description, sequence))  # Store description and sequence

    # Extract overall_confidence from description
    for i, (desc, seq) in enumerate(sequences):
        # Use regex to extract overall_confidence value
        match = re.search(r'overall_confidence=([0-9.]+)', desc)
        if match:
            overall_confidence = float(match.group(1))
            sequences[i] = (seq, overall_confidence)  # Update to sequence and overall_confidence

    return sequences[1:]  # Skip the first entry if needed


import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, Superimposer
from Bio.PDB.Atom import Atom
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from Bio.PDB import MMCIFParser, PDBParser, Superimposer, PDBIO
import numpy as np
import os
from collections import defaultdict
import tempfile
from Bio.PDB import Structure, Model, Chain,Residue

def get_ligands(structure):
    """Return a dictionary of ligands grouped by residue ID {res_id: list of atoms}"""
    ligands = defaultdict(list)
    for res in structure.get_residues():
        if not is_amino_acid(res) and not is_nucleic_acid(res) and res.get_resname().strip() != 'HOH':
            res_id = (res.parent.id, res.id[1], "LIG")  # (chain ID, residue number, residue name)
            ligands[res_id].extend([atom for atom in res.get_atoms() if atom.element != 'H'])
    return ligands

def save_ligand_residue(residue_atoms, output_file):
    """Save a single ligand residue to a temporary file"""
    structure = Structure.Structure('temp')
    model = Model.Model(0)
    chain = Chain.Chain('A')
    new_res = Residue.Residue(id=(' ', 0, ' '), resname='LIG', segid='')
    for atom in residue_atoms:
        new_res.add(atom.copy())
    chain.add(new_res)
    model.add(chain)
    structure.add(model)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)

def calculate_multi_ligand_rmsd(cif_ligands, pdb_ligands):
    ligand_rmsds = []
    atom_distances = []
    
    # Find ligands with matching residue IDs

    common_keys = set(cif_ligands.keys()) & set(pdb_ligands.keys())
    if not common_keys:
        raise ValueError("No common keys found between cif_ligands and pdb_ligands. maybe you need to reindex atoms")
    for key in common_keys:
        try:
            # Create a temporary file

            with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as cif_temp, \
                 tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as pdb_temp:
                
                save_ligand_residue(cif_ligands[key], cif_temp.name)
                save_ligand_residue(pdb_ligands[key], pdb_temp.name)
 
                # Get the reference SMILES

                ref_mol = Chem.MolFromPDBFile(pdb_temp.name, removeHs=True)
                if not ref_mol: continue

                ref_smiles = Chem.MolToSmiles(ref_mol, canonical=True)
                
                # Calculate RMSD

                dist, rmsd, _ = align_molecules_from_pdb_files(
                    ref_smiles, cif_temp.name, pdb_temp.name

                )
                ligand_rmsds.append(rmsd)
                atom_distances.append(dist)
                
            # Clean up temporary files

            os.unlink(cif_temp.name)
            os.unlink(pdb_temp.name)
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
    
    return {
        'ligand_rmsd_mean': np.mean(ligand_rmsds) if ligand_rmsds else None,
        'ligand_rmsd_list': ligand_rmsds,
        'atom_distances': atom_distances

    }

def get_ca_atoms(structure):
    """Get a list of Cα atoms sorted by chain ID and residue number"""
    ca_atoms = []
    for model in structure:
        # Sort chains by chain ID
        for chain in sorted(model, key=lambda c: c.id):
            # Sort residues by residue number
            for residue in sorted(chain, key=lambda r: r.id[1]):
                if 'CA' in residue:
                    ca_atoms.append(residue['CA'])
    return ca_atoms


def get_small_molecules(structure):
    """Get all non-protein, non-nucleic acid, non-water heavy atoms"""
    return [
        atom
        for residue in structure.get_residues()
        # Exclude standard residue types
        if not is_amino_acid(residue)
        and not is_nucleic_acid(residue)
        and residue.get_resname().strip() != 'HOH'
        # Iterate over atoms and filter out hydrogens
        for atom in residue.get_atoms()
        if atom.element != 'H'
    ]


def is_amino_acid(residue):
    return residue.get_resname().strip() in {
        'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
        'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'}

def is_nucleic_acid(residue):
    return residue.get_resname().strip() in {'DA','DT','DC','DG','A','U','C','G'}


def save_ligand_structure(structure, output_file):
    io = PDBIO()
    
    from Bio.PDB import Structure, Model, Chain, Residue
    new_structure = Structure.Structure('ligand_only')
    new_model = Model.Model(0)  # new Model
    new_chain = Chain.Chain('A')  # new Chain
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() not in [
                    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 
                    'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 
                    'THR', 'VAL', 'TRP', 'TYR', 'HOH'
                ]:
                    new_residue = Residue.Residue((' ', 1, ' '), 'LIG', '')
                    
                    # Copy the original non-hydrogen atoms to a new residue
                    for atom in residue:
                        if atom.element != 'H':  # exculde H
                            new_residue.add(atom.copy())

                    new_chain.add(new_residue)
    
    # Assemble the new structure

    new_model.add(new_chain)
    new_structure.add(new_model)
    
    # Save the new structure
    io.set_structure(new_structure)
    io.save(output_file)


def read_pdb_coordinates(pdb_file):
    """
    Read atomic coordinates from a PDB file
    """

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
    """
    Calculate the RMSD of two coordinate sets.
    """
    diff = []
    atom1 =[]
    atom2 =[]
    for idx1, idx2 in atom_map:
        # Use atom name to index coordinate dictionary
        atom1_name = list(coords1.keys())[idx1]
        atom2_name = list(coords2.keys())[idx2]
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
    # Create reference molecule from SMILES
    ref_mol = Chem.MolFromSmiles(smiles)
    if ref_mol is None:
        raise ValueError("Could not create reference molecule from SMILES")
    
    # Read molecule from PDB file
    mol1 = Chem.MolFromPDBFile(pdb_file1, removeHs=True)
    mol2 = Chem.MolFromPDBFile(pdb_file2, removeHs=True)
    
    if mol1 is None or mol2 is None:
        raise ValueError("Could not create molecule from PDB file")

    # Get canonical SMILES to check for structural consistency
    mol1_smiles = Chem.MolToSmiles(mol1, canonical=True)
    mol2_smiles = Chem.MolToSmiles(mol2, canonical=True)
    print(f"mol1 SMILES: {mol1_smiles}")
    print(f"mol2 SMILES: {mol2_smiles}")

    # Molecular matching
    matches1 = mol1.GetSubstructMatch(ref_mol)
    matches2 = mol2.GetSubstructMatch(ref_mol)
    
    # Generate atom map
    atom_map = [(idx1, idx2) for idx1, idx2 in zip(matches1, matches2)]

    if not atom_map:
        raise ValueError("Atom mapping is empty!")

    # Read coordinates from PDB file
    coords1 = read_pdb_coordinates(pdb_file1)
    coords2 = read_pdb_coordinates(pdb_file2)

    # RMSD
    rmsd,atom_distance = calculate_rmsd_from_coordinates(coords1, coords2, atom_map)
    return atom_distance, rmsd, atom_map


def calculate_atomic_rmsd(ref_atoms, model_atoms):
    if len(ref_atoms) != len(model_atoms):
        return 0.0

    ref_coords = np.array([atom.get_coord() for atom in ref_atoms])
    model_coords = np.array([atom.get_coord() for atom in model_atoms])
    return np.sqrt(np.mean(np.sum((ref_coords - model_coords)**2, axis=1)))

def get_nucleic_acids(structure, na_type):
    na_res = {'DNA': ['DA','DT','DC','DG'], 'RNA': ['A','U','C','G']}[na_type]
    atoms = []
    for residue in structure.get_residues():
        if residue.get_resname().strip() in na_res:
            # Extract phosphate backbone atoms (P, O4', C4'), ensuring consistent atom order

            sorted_atoms = sorted(residue.get_atoms(), key=lambda a: a.get_name())
            atoms.extend([atom for atom in sorted_atoms if atom.get_name() in ['O4\'','C4\'']])
    return atoms

def get_ca_dict(structure,fixed_chains=[]):
    """Returns a dictionary of {(chain ID, residue number): Cα atom} for precise matching"""
    ca_dict = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if fixed_chains != []:
                if chain_id in fixed_chains:
                    for residue in chain:
                        res_id = residue.id[1]  
                        if 'CA' in residue:
                            ca_dict[(chain_id, res_id)] = residue['CA']
            else:
                for residue in chain:
                    res_id = residue.id[1] 
                    if 'CA' in residue:
                        ca_dict[(chain_id, res_id)] = residue['CA']
    return ca_dict

def remove_inter_chain_connections(structure):
    """Remove connections between different chains in a structure."""
    for model in structure:
        for chain in model:
            # Collect all atoms in the chain

            chain_atoms = {atom.get_serial_number() for residue in chain for atom in residue}
            for residue in chain:
                for atom in residue:
                    # Filter out bonds that include atoms not in the same chain

                    atom.bonds = [bonded_atom for bonded_atom in atom.bonds if bonded_atom.get_serial_number() in chain_atoms]
    return structure

def calculate_ca_rmsd(cif_file, pdb_file,fixed_chains=None):
    protein_rmsd = 100.0
    binder_rmsd = 100.0
    ligand_rmsd = 100.0

    dna_rmsd = 100.0

    rna_rmsd = 100.0

    atom_centre_distance = 100.0
    ligand_results = None
    try:
        # Parse file
        cif_parser = MMCIFParser(QUIET=True)
        cif_structure = cif_parser.get_structure("CIF", cif_file)
        #cif_structure = remove_inter_chain_connections(cif_structure)
        
        pdb_parser = PDBParser(QUIET=True)
        pdb_structure = pdb_parser.get_structure("PDB", pdb_file)
        #pdb_structure = remove_inter_chain_connections(pdb_structure)
        # get Cα atom
        if fixed_chains:
            cif_ca_dict_fix = get_ca_dict(cif_structure,fixed_chains)
            pdb_ca_dict_fix = get_ca_dict(pdb_structure,fixed_chains)
            # Find common (chain ID, residue number)
            common_keys = set(cif_ca_dict_fix.keys()) & set(pdb_ca_dict_fix.keys())
            if not common_keys:
                raise ValueError("CIF and PDB files have no common Ca atoms for RMSD calculation")
            sorted_keys = sorted(common_keys)  # Sort by chain ID and residue number
            cif_ca = [cif_ca_dict_fix[key] for key in sorted_keys]
            pdb_ca = [pdb_ca_dict_fix[key] for key in sorted_keys]
    
            # Extract coordinates and align
            super_imposer = Superimposer()
            super_imposer.set_atoms(cif_ca, pdb_ca) 
            super_imposer.apply(pdb_structure.get_atoms())  # Modify PDB structure coordinates
    
            protein_rmsd = super_imposer.rms

            cif_ca_dict_all = get_ca_dict(cif_structure)
            pdb_ca_dict_all = get_ca_dict(pdb_structure)

            # Find common (chain ID, residue number)
            common_keys_all = (set(cif_ca_dict_all.keys()) & set(pdb_ca_dict_all.keys()))-common_keys
            sorted_keys_all = sorted(common_keys_all)  # Sort by chain ID and residue number

            cif_ca_all = [cif_ca_dict_all[key] for key in sorted_keys_all]
            pdb_ca_all = [pdb_ca_dict_all[key] for key in sorted_keys_all]
            super_imposer_all = Superimposer()
            super_imposer_all.set_atoms(cif_ca_all, pdb_ca_all) 
            binder_rmsd = super_imposer_all.rms
        
        else:
            cif_ca_dict_all = get_ca_dict(cif_structure)
            pdb_ca_dict_all = get_ca_dict(pdb_structure)

            # Find common (chain ID, residue number)
            common_keys = set(cif_ca_dict_all.keys()) & set(pdb_ca_dict_all.keys())
            if not common_keys:
                raise ValueError("CIF and PDB files have no common Ca atoms for RMSD calculation")
            sorted_keys = sorted(common_keys) 
            cif_ca = [cif_ca_dict_all[key] for key in sorted_keys]
            pdb_ca = [pdb_ca_dict_all[key] for key in sorted_keys]

            # Extract coordinates and align
            super_imposer = Superimposer()
            super_imposer.set_atoms(cif_ca, pdb_ca) 
            super_imposer.apply(pdb_structure.get_atoms())  # Modify PDB structure coordinates

            protein_rmsd = super_imposer.rms
            binder_rmsd = super_imposer.rms
        
        # Process ligand
        cif_ligands = get_ligands(cif_structure)
        pdb_ligands = get_ligands(pdb_structure)
        
        if cif_ligands and pdb_ligands:
            if len(cif_ligands) != len(pdb_ligands):
                print(len(cif_ligands) , len(pdb_ligands))
                print("Small molecule atom counts mismatch, skipping small molecule RMSD calculation")

            ligand_results = calculate_multi_ligand_rmsd(cif_ligands, pdb_ligands)
            
        else:
            print("no ligand atom")
            ligand_results = None
        cif_dna = get_nucleic_acids(cif_structure, 'DNA')
        pdb_dna = get_nucleic_acids(pdb_structure, 'DNA')
            
        if cif_dna and pdb_dna:
            if len(cif_dna) == len(pdb_dna):
                dna_rmsd = calculate_atomic_rmsd(cif_dna, pdb_dna)
            else:
                print(f"DNA atom counts mismatch CIF: {len(cif_dna)}, PDB: {len(pdb_dna)}")

        cif_rna = get_nucleic_acids(cif_structure, 'RNA')
        pdb_rna = get_nucleic_acids(pdb_structure, 'RNA')
        if cif_rna and pdb_rna:
            if len(cif_rna) == len(pdb_rna):
                rna_rmsd = calculate_atomic_rmsd(cif_rna, pdb_rna)
            else:
                print(f"RNA atom counts mismatch CIF: {len(cif_rna)}, PDB: {len(pdb_rna)}")
    except Exception as e:
        print(f"Error during RMSD calculation: {e}")
    if ligand_results:
        return {
            'protein_rmsd': [binder_rmsd,protein_rmsd],
            'ligand_rmsd': ligand_results['ligand_rmsd_list'],
            'ligand_count': len(ligand_results['ligand_rmsd_list']),
            'atom_distances': ligand_results['atom_distances'],
            'dna_rmsd': dna_rmsd,
            'rna_rmsd': rna_rmsd
        }
    else:
        return {
            'protein_rmsd': [binder_rmsd,protein_rmsd],
            'ligand_rmsd': 100,
            'ligand_count': 1,
            'atom_distances': 100,
            'dna_rmsd': dna_rmsd,
            'rna_rmsd': rna_rmsd
        }

def clear_gpu_memory():
    """Clear GPU memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
def read_file(path: pathlib.Path, json_path: Optional[pathlib.Path]) -> str:
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


def is_protein_chain(chain):
    """Check if a given chain is a protein chain by looking for amino acids."""
    for residue in chain:
        if residue.id[0] == " " and residue.resname in protein_residues:
            return True

    return False

protein_residues = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU',
    'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
    'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

def split_cif_by_chain(input_cif_path, output_dir,fixed_chains):
    os.makedirs(output_dir, exist_ok=True)

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", input_cif_path)

    out_path_list = []
    for model in structure:
        for chain in model:
            if is_protein_chain(chain) and chain.id not in fixed_chains:
                # Create a new structure for the chain

                chain_structure = model.__class__(model.id)
                chain_structure.add(chain.copy())

                # Save the new chain structure to a CIF file

                output_cif_path = os.path.join(output_dir, f"chain_{chain.id}.cif")
                io = MMCIFIO()
                io.set_structure(chain_structure)
                io.save(output_cif_path)
                out_path_list.append(output_cif_path)
    return out_path_list

# Define protein residues to check if a chain is a protein chain

from Bio import PDB

def reindex_structure(input_file, output_file):
    # Create a PDB parser
    
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', input_file)

    # Iterate over each model, chain, and residue
    
    for model in structure:
        for chain in model:
            new_resid = 1

            for residue in chain:
                # Get the current residue ID
                
                old_id = residue.id

                # Create a new ID, preserving the original resname and icode
                
                new_id = (' ', new_resid, old_id[2])
                # Update the residue ID
                
                residue.id = new_id

                new_resid += 1

    # Write the modified structure to a file using PDBIO
    
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)


def calculate_average_b_factor(cif_file: str, chain_id: list = ["A"]) -> float:
    """
    Calculate the average B-factor for a specified chain in a CIF file.

    Parameters:
        cif_file (str): Path to the CIF file.
        chain_id (str): ID of the chain for which to calculate the B-factor, defaults to "A".

    Returns:
        float: The average B-factor of the specified chain, or 0.0 if the chain does not exist or has no atoms.
    """
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file)
        
        b_factors = []
        for model in structure:
            for chain in model:
                if chain.id in chain_id:  
                    for residue in chain:
                        for atom in residue:
                            b_factors.append(atom.bfactor)
        
        return sum(b_factors) / len(b_factors) if b_factors else 0.0
    except Exception as e:
        print(f"B-factor calculation failed: {str(e)}")
        return 0.0

def calculate_key_b_factor(cif_file: str, cutoff=5) -> float:
    """
    Calculate the average B-factor for a specified chain in a CIF file.

    Parameters:
        cif_file (str): Path to the CIF file.
        chain_id (str): ID of the chain for which to calculate the B-factor, defaults to "A".

    Returns:
        float: The average B-factor of the specified chain, or 0.0 if the chain does not exist or has no atoms.
    """
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file)
    
        sm_atoms = []
        nearby_atoms = []

        for model in structure:
            for chain in model:
                if chain.id in ["A"]:  
                    for residue in chain:
                        for atom in residue:
                            nearby_atoms.append((residue.id, atom.serial_number, atom))  # Using residue and atom indices
                else:
                    for residue in chain:
                        for atom in residue:
                            sm_atoms.append((residue.id, atom.serial_number, atom))  # Using residue and atom indices


        # Calculate atoms around small molecule
        nearby_atoms_within_cutoff = []
        for sm_res_id, sm_atom_id, sm_atom in sm_atoms:
            for res_id, atom_id, atom in nearby_atoms:
                distance = sm_atom - atom
                if distance <= cutoff:
                    nearby_atoms_within_cutoff.append((res_id, atom_id, atom))
        
        b_factors_with_ligand = [atom.bfactor for _, _, atom in nearby_atoms_within_cutoff]
        return sum(b_factors_with_ligand) / len(b_factors_with_ligand) if b_factors_with_ligand else 0.0
    except Exception as e:
        print(f"B-factor calculation failed: {str(e)}")
        return 0.0


def find_pocket_residues_based_on_distance(pdbfile: str, cutoff=5.0) -> list:
    """
    Find protein residues in a PDB file that are within a specified cutoff distance of all non-protein residues.

    Parameters:
        pdbfile (str): Path to the PDB file.
        cutoff (float): Distance threshold (Å).
    
    Returns:
        list: A list of residues in the format ["A1", "B5"].
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("target", pdbfile)
        
        sm_atoms = []       # Store all non-protein atoms
        protein_atoms = []  # Store all protein atoms
        nearby_residues = set() # Store nearby residues

        # Standard amino acid three-letter codes (manually defined)
        STANDARD_AMINO_ACIDS = set([aa.upper() for aa in aa3])

        # Iterate over all atoms in the structure

        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.resname.strip().upper()
                    is_protein = resname in STANDARD_AMINO_ACIDS

                    for atom in residue:
                        if all(coord == 0.0 for coord in atom.coord):
                            continue

                            
                        if is_protein:
                            protein_atoms.append(atom)
                        else:
                            sm_atoms.append(atom)

        # Debug information: check atom counts
        print(f"[DEBUG] Detected protein atom count: {len(protein_atoms)}, small molecule atom count: {len(sm_atoms)}")

        # Return early if no small molecule or protein atoms are found
        if not sm_atoms or not protein_atoms:
            print("Error: No small molecule or protein atoms detected")
            return []

        # Build NeighborSearch object (including only protein atoms)
        ns = NeighborSearch(protein_atoms)
        
        # Find nearby protein residues

        for atom in sm_atoms:
            close_atoms = ns.search(atom.coord, cutoff)
            for nearby_atom in close_atoms:
                residue = nearby_atom.get_parent()
                chain = residue.get_parent()
                
                # Handle residue numbering (supports insertion codes like 100A)
                res_id = str(residue.id[1])
                if residue.id[2].strip():  

                    res_id += residue.id[2]
                
                residue_code = f"{chain.id}{res_id}"
                nearby_residues.add(residue_code)

        # Custom sorting function (sort by chain ID first, then by the numeric part)
        def sort_key(code):
            chain_part = code[0]
            num_part = ''.join(filter(str.isdigit, code[1:])) or '0'
            return (chain_part, int(num_part))

        return sorted(nearby_residues, key=sort_key)

    except Exception as e:
        print(f"Error: {str(e)}")
        return []


def parse_residue_string(res_str):
    """Parse a residue string (e.g., 'A1', 'B100A') into chain ID, sequence number, and insertion code"""
    match = re.match(r'^([A-Za-z])(\d+)([A-Za-z]?)$', res_str, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid residue format: {res_str}")
    chain_id = match.group(1).upper()
    seq_num = int(match.group(2))
    icode = match.group(3).upper() if match.group(3) else ''
    return (chain_id, seq_num, icode)

def calculate_plddt_avg(cif_file, residue_list):
    # Parse target residue list

    target_residues = set()
    for res_str in residue_list:
        try:
            target = parse_residue_string(res_str)
            target_residues.add(target)
        except ValueError as e:
            print(f"Skipping invalid residue: {e}")
            continue

    
    # Prepare CIF parser

    parser = MMCIFParser()
    structure = parser.get_structure('protein', cif_file)
    
    # Collect pLDDT values

    plddt_avgs = []
    for model in structure:
        for chain in model:
            chain_id = chain.id.upper()
            for residue in chain:
                res_id = residue.id

                seq_num = res_id[1]
                icode = res_id[2].strip().upper()
                
                if (chain_id, seq_num, icode) in target_residues:
                    b_factors = []
                    for atom in residue:
                        if atom.bfactor is not None:
                            b_factors.append(atom.bfactor)
                    if b_factors:
                        avg = sum(b_factors) / len(b_factors)
                        plddt_avgs.append(avg)
                    else:
                        print(f"Warning: Residue {chain_id}{seq_num}{icode or ''} has no valid atom data")
    
    if not plddt_avgs:
        global_avg= 0
    else:
        global_avg = sum(plddt_avgs) / len(plddt_avgs)
    return global_avg

def common_elements(*lists):
    """
    Find common elements in multiple lists.

    Parameters:
    *lists : multiple list arguments

    Returns:
    A list containing the common elements.
    """
    if not lists:
        return []

    # Convert each list to a set

    sets = map(set, lists)

    # Find the intersection of all sets

    common_set = set.intersection(*sets)

    # Convert the set back to a list if needed

    return list(common_set)


from Bio.PDB import MMCIFParser

import numpy as np

def calculate_bfactor_averages_from_list(cif_file, residue_list):
    """
    Calculate the average all-atom B-factor for residues both inside and outside 
    a specified list from a CIF file.
    
    :param cif_file: str, Path to the CIF file.
    :param residue_list: list, Strings formatted as chain ID + residue number, e.g., ["A1", "B2", "C3"].
    :return: tuple, (average B-factor inside list, average B-factor outside list).
    """
    def parse_residue_list(residue_list):
        """
        Parse the input amino acid list, e.g., ['A1', 'B2', 'C3'].
        :param residue_list: list, Strings formatted as chain ID + residue number, e.g., ['A1', 'B2', 'C3']
        :return: dict, Keys are chain IDs, values are sets of residue numbers.
        """
        parsed = {}
        for item in residue_list:
            chain_id = item[0]  # Chain ID (e.g., 'A')
            residue_id = int(item[1:])  # Residue number (e.g., 1, 2, 3)
            if chain_id not in parsed:
                parsed[chain_id] = set()
            parsed[chain_id].add(residue_id)
        return parsed

    # Parse CIF file

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_file)

    # Parse amino acid list

    parsed_residues = parse_residue_list(residue_list)

    # Initialize variables

    bfactor_in = []
    bfactor_out = []

    # Iterate over each amino acid residue

    for model in structure:
        for chain in model:
            chain_id = chain.id  

            for residue in chain:
                # Skip water molecules or other HETATMs 

                if residue.id[0] != " ":
                    continue

                residue_id = residue.id[1]  

                is_in_list = chain_id in parsed_residues and residue_id in parsed_residues[chain_id]

                # Iterate over all atoms within the residue

                for atom in residue.get_atoms():
                    if is_in_list:
                        bfactor_in.append(atom.bfactor)
                    else:
                        bfactor_out.append(atom.bfactor)

    avg_bfactor_in = np.mean(bfactor_in) if bfactor_in else None

    avg_bfactor_out = np.mean(bfactor_out) if bfactor_out else None

    return avg_bfactor_in, avg_bfactor_out