import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
import os
import pickle
import argparse
from collections import Counter
from datetime import datetime
import logging
import numpy as np
from Bio.PDB import PDBIO
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile
import os

from Bio.PDB import Structure, Model, Chain
# Atom mapping directly
DENSE_ATOM = {
    # Protein
    "ALA": ("N", "CA", "C", "O", "CB"),
    "ARG": ("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    "GLY": ("N", "CA", "C", "O"),
    "HIS": ("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    "PHE": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD"),
    "SER": ("N", "CA", "C", "O", "CB", "OG"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    "TRP": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
    "TYR": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    "UNK": (),
    # RNA
    "A": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
          "C2PRIME", "O2PRIME", "C1PRIME", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
    "C": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
          "C2PRIME", "O2PRIME", "C1PRIME", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "G": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
          "C2PRIME", "O2PRIME", "C1PRIME", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
    "U": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
          "C2PRIME", "O2PRIME", "C1PRIME", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
    "UNK_RNA": (),
    # DNA
    "DA": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
           "C2PRIME", "C1PRIME", "N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"),
    "DC": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
           "C2PRIME", "C1PRIME", "N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
    "DG": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
           "C2PRIME", "C1PRIME", "N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"),
    "DT": ("OP3", "P", "OP1", "OP2", "O5PRIME", "C5PRIME", "C4PRIME", "O4PRIME", "C3PRIME", "O3PRIME",
           "C2PRIME", "C1PRIME", "N1", "C2", "O2", "N3", "C4", "O4", "C5", "C7", "C6"),
    "UNK_DNA": (),
}
_BUCKETS  = (
    256,
    512,
    768,
    1024,
    1280,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
    4608,
    5120,
)
def pad_array_to_bucket(array, buckets=_BUCKETS):
    """
    Pads the input array to the smallest bucket size that can accommodate its second dimension.
    
    Args:
        array (np.ndarray): Input array with shape (1, L, 24, 3).
        buckets (tuple): A tuple of bucket sizes.
    
    Returns:
        np.ndarray: Padded array with the target bucket size.
        int: The bucket size used for padding.
    """
    # Get the current shape of the input array
    current_shape = array.shape
    L = current_shape[1]  # Second dimension (sequence length)

    # Find the smallest bucket that can fit the current array
    target_bucket = next((bucket for bucket in buckets if bucket >= L), None)
    
    if target_bucket is None:
        raise ValueError(f"No bucket can accommodate sequence length {L}. Consider increasing bucket sizes.")

    # Calculate padding sizes for the second dimension (L to target_bucket)
    padding = [(0, 0),  # No padding for the first dimension (batch size)
               (0, target_bucket - L),  # Pad to target_bucket in the second dimension
               (0, 0),  # No padding for the third dimension
               (0, 0)]  # No padding for the fourth dimension
    
    # Apply padding with constant value 0
    padded_array = np.pad(array, padding, mode='constant', constant_values=0)
    
    return padded_array, target_bucket

def is_nucleic(residue_name):
    """Check if residue is DNA/RNA"""
    return residue_name in {'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT'}

def is_protein(residue_name):
    """Check if residue is a protein"""
    protein_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }
    return residue_name in protein_residues

def estimate_missing_atom(residue, atom_name, reference_atom, bond_length, direction=None):
    """
    Estimate the position of a missing atom based on a reference atom and bond length.
    """
    if reference_atom in residue:
        ref_atom_coord = residue[reference_atom].coord
        if direction is None:
            direction = np.array([1.0, 0.0, 0.0])
        
        direction = direction / np.linalg.norm(direction)
        estimated_coord = ref_atom_coord + bond_length * direction
        return estimated_coord
    else:
        raise ValueError(f"Reference atom {reference_atom} not found in residue.")



def export_residue_to_pdb(residue):
    """
    Export a residue to a temporary PDB file by manually creating a minimal Structure object.
    """
    try:
        # Create a minimal Structure object
        structure = Structure.Structure("minimal_structure")
        model = Model.Model(0)  # Model ID
        chain = Chain.Chain("A")  # Chain ID
        chain.add(residue)  # Add the residue to the chain
        model.add(chain)  # Add the chain to the model
        structure.add(model)  # Add the model to the structure

        # Save the minimal structure to a temporary PDB file
        io = PDBIO()
        io.set_structure(structure)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
        io.save(temp_file.name)

        return temp_file.name

    except Exception as e:
        print(f"Error exporting residue to PDB: {e}")
        return None

def align_and_reorder_atoms(pdb_mol, ref_mol):
    """
    Align PDB molecule with SMILES molecule and return the atom mapping.
    """
    try:
        # Compute the Maximum Common Substructure (MCS)
        mcs = Chem.MolFromSmarts(Chem.MolToSmarts(pdb_mol))
        if mcs is None:
            print("Error: MCS could not be computed.")
            return None

        # Get the atom indices of the MCS in both molecules
        mcs_atoms_pdb = pdb_mol.GetSubstructMatch(mcs)
        mcs_atoms_ref = ref_mol.GetSubstructMatch(mcs)
        if not mcs_atoms_pdb or not mcs_atoms_ref:
            mol1_smiles = Chem.MolToSmiles(pdb_mol, canonical=True)
            mol2_smiles = Chem.MolToSmiles(ref_mol, canonical=True)
            print(f"PDB Molecule SMILES: {mol1_smiles}")
            print(f"Reference Molecule SMILES: {mol2_smiles}")
            print(mcs_atoms_pdb , mcs_atoms_ref)
            print("Error: No substructure match found.")
            return None

        # Create a mapping between the atoms in the PDB molecule and the reference molecule
        atom_mapping = dict(zip(mcs_atoms_pdb, mcs_atoms_ref))
        return atom_mapping

    except Exception as e:
        print(f"Error aligning and reordering atoms: {e}")
        return None

def reorder_array(array, atom_mapping, pdb_atoms):
    """
    Reorder the array based on the atom mapping.
    """
    try:
        if not atom_mapping:
            print("Error: Atom mapping is empty.")
            return None

        print(f"Array shape: {array.shape}")
        print(f"Atom mapping: {atom_mapping}")
        print(f"PDB atoms: {[atom.name for atom in pdb_atoms]}")

        reordered_array = np.zeros_like(array)
        for pdb_idx, ref_idx in atom_mapping.items():
            print(f"Processing atom mapping: pdb_idx={pdb_idx}, ref_idx={ref_idx}")
            if pdb_idx >= len(pdb_atoms) or ref_idx >= array.shape[1]:
                print(f"Error: Invalid atom mapping index. pdb_idx={pdb_idx}, ref_idx={ref_idx}")
                return None
            pdb_atom = pdb_atoms[pdb_idx]
            pdb_atom_idx = pdb_atoms.index(pdb_atom)
            print(f"PDB atom: {pdb_atom.name}, pdb_atom_idx={pdb_atom_idx}")
            reordered_array[0, ref_idx, 0] = array[0, pdb_atom_idx, 0]
            print(f"Reordered array at [0, {ref_idx}, 0]: {reordered_array[0, ref_idx, 0]}")

        print(f"Reordered array shape: {reordered_array.shape}")
        return reordered_array

    except Exception as e:
        print(f"Error reordering array: {e}")
        return None

def process_small_molecule(residue):
    """
    Process small molecule residue, align with a reference molecule, and reorder atoms.
    Returns an array of reordered atom coordinates.
    """
    # Step 1: Export the residue to a temporary PDB file
    pdb_file = export_residue_to_pdb(residue)
    if pdb_file is None:
        return None

    try:
        # Step 2: Extract atom coordinates from the residue and count non-hydrogen atoms
        atoms = list(residue.get_atoms())
        atoms = [atom for atom in atoms if atom.element != 'H']
        if not atoms:
            print("Error: No non-hydrogen atoms found in the residue.")
            return None

        num_atoms = len(atoms)
        array = np.zeros((1, num_atoms, 1, 3))  # [1, num_atoms, 1, 3] shape

        for i, atom in enumerate(atoms):
            array[0, i, 0] = atom.coord
            print(f"Atom {i}: {atom.name} - Coordinates: {atom.coord}")

        # Step 3: Read the PDB file and prepare RDKit molecule representations
        pdb_mol = Chem.MolFromPDBFile(pdb_file, removeHs=True)
        if pdb_mol is None:
            print("Error reading PDB file into RDKit molecule.")
            return None

        mol1_smiles = Chem.MolToSmiles(pdb_mol, canonical=True,doRandom=False, isomericSmiles=False)
        ref_mol = Chem.MolFromSmiles(mol1_smiles)
        if ref_mol is None:
            print("Error generating reference molecule from SMILES.")
            return None

        # Step 4: Align and reorder atoms
        atom_mapping = align_and_reorder_atoms(pdb_mol, ref_mol)
        if atom_mapping is None:
            return None

        # Step 5: Reorder the array based on the atom mapping
        reordered_array = reorder_array(array, atom_mapping, atoms)
        if reordered_array is None:
            return None

    except Exception as e:
        print(f"Error processing small molecule: {e}")
        return None

    finally:
        # Step 6: Delete the temporary file
        if os.path.exists(pdb_file):
            try:
                os.remove(pdb_file)
            except Exception as e:
                print(f"Error deleting temporary file: {e}")

    # Return the reordered array
    return reordered_array

def process_protein_residue(residue, array, pos, dense_atom_mapping):
    """Process protein residue and handle terminal modifications."""
    residue_name = residue.get_resname()
    atom_names = dense_atom_mapping.get(residue_name, dense_atom_mapping["UNK"])
    
    for j, atom_name in enumerate(atom_names):
        if atom_name in residue and residue[atom_name].coord is not None:
            array[0, pos, j, :] = residue[atom_name].coord

        else:
            print(f"Warning: Atom '{atom_name}' not found or has no coordinates in residue '{residue_name}'")
            array[0, pos, j, :] = [0.0,0.0,0.0]

    return array

def process_nucleic_residue(residue, array, pos, dense_atom_mapping):
    """Process DNA/RNA residue."""
    residue_name = residue.get_resname()
    if residue_name in dense_atom_mapping:
        atom_names = dense_atom_mapping[residue_name]
    else:
        atom_names = dense_atom_mapping["UNK_RNA"] if "UNK_RNA" in dense_atom_mapping else dense_atom_mapping["UNK_DNA"]
    
    for j, atom_name in enumerate(atom_names):
        atom_name = atom_name.replace("PRIME","'")
        if atom_name in residue and residue[atom_name].coord is not None:
            array[0, pos, j, :] = residue[atom_name].coord

        else:
            #array[0, pos, j, :] = [0,0,0]
            print(f"Warning: Atom '{atom_name}' not found or has no coordinates in residue '{residue_name}'")
            array[0, pos, j, :] = [0.0,0.0,0.0]
    return array

def process_chain(chain, dense_atom_mapping):
    """
    Process a single chain and return appropriate array based on chain type.
    """
    residues = list(chain.get_residues())
    if not residues:
        return None, None
        
    # Check chain type based on first residue
    first_residue = residues[0]
    residue_name = first_residue.get_resname()
    # If it's a small molecule chain
    if not (is_protein(residue_name) or is_nucleic(residue_name)):
        array = process_small_molecule(first_residue)
        return array, "small_molecule"
    
    # For protein or nucleic acid chains
    L = len(residues)
    max_atoms = 24  # Maximum number of atoms per residue
    array = np.zeros((1, L, max_atoms, 3))
    
    for i, residue in enumerate(residues):
        residue_name = residue.get_resname()
        if is_protein(residue_name):
            array = process_protein_residue(residue, array, i, dense_atom_mapping)
            
            # Handle protein termini
            if residue.get_id()[0] == ' ':
                if i == 0:  # N-terminus
                    try:
                        if 'NH3' not in residue:
                            nh3_coord = estimate_missing_atom(
                                residue, 'NH3', reference_atom='N', bond_length=1.02,
                                direction=np.array([0.0, 0.0, -1.0])
                            )
                            array[0, i, 23, :] = nh3_coord
                    except ValueError as e:
                        print(f"Warning: {e}")
                
                if i == L - 1:  # C-terminus
                    try:
                        if 'OXT' not in residue:
                            oxt_coord = estimate_missing_atom(
                                residue, 'OXT', reference_atom='C', bond_length=1.25,
                                direction=np.array([0.0, 0.0, 1.0])
                            )
                            array[0, i, 23, :] = oxt_coord
                    except ValueError as e:
                        print(f"Warning: {e}")
                        
        elif is_nucleic(residue_name):
            array = process_nucleic_residue(residue, array, i, dense_atom_mapping)
    
    chain_type = "protein" if is_protein(residue_name) else "nucleic"
    return array, chain_type

def parse_structure_to_array(file_path, dense_atom_mapping):
    """
    Enhanced parser that handles proteins, DNA, RNA, and small molecules.
    
    Args:
        file_path: Path to structure file
        dense_atom_mapping: Dictionary mapping residue names to atom names
        
    Returns:
        tuple: (array of coordinates, molecule_type)
    """
    parser = MMCIFParser(QUIET=True) if file_path.endswith('.cif') else PDBParser(QUIET=True)
    structure = parser.get_structure("molecule", file_path)
    
    # First, determine molecule type and count residues
    residues = list(structure.get_residues())
    if not residues:
        raise ValueError("No residues found in structure")
        
    first_residue = residues[0]
    residue_name = first_residue.get_resname()
    
    # Handle small molecules
    if not (is_protein(residue_name) or is_nucleic(residue_name)):
        return process_small_molecule(first_residue)
    
    # Handle proteins and nucleic acids
    L = len(residues)
    max_atoms = 24  # Maximum number of atoms per residue
    array = np.zeros((1, L, max_atoms, 3))
    
    for i, residue in enumerate(residues):
        residue_name = residue.get_resname()
        
        if is_protein(residue_name):
            array = process_protein_residue(residue, array, i, dense_atom_mapping)
            
            # Handle protein termini
            if residue.get_id()[0] == ' ':
                if i == 0:  # N-terminus
                    try:
                        if 'NH3' not in residue:
                            nh3_coord = estimate_missing_atom(
                                residue, 'NH3', reference_atom='N', bond_length=1.02,
                                direction=np.array([0.0, 0.0, -1.0])
                            )
                            array[0, i, 23, :] = nh3_coord
                    except ValueError as e:
                        print(f"Warning: {e}")
                
                if i == L - 1:  # C-terminus
                    try:
                        if 'OXT' not in residue:
                            oxt_coord = estimate_missing_atom(
                                residue, 'OXT', reference_atom='C', bond_length=1.25,
                                direction=np.array([0.0, 0.0, 1.0])
                            )
                            array[0, i, 23, :] = oxt_coord
                    except ValueError as e:
                        print(f"Warning: {e}")
                        
        elif is_nucleic(residue_name):
            array = process_nucleic_residue(residue, array, i, dense_atom_mapping)
            
    return array, "protein" if is_protein(residues[0].get_resname()) else "nucleic"

def estimate_missing_atom(residue, atom_name, reference_atom, bond_length, direction=None):
    """
    Estimate the position of a missing atom based on a reference atom and bond length.

    Args:
        residue (Bio.PDB.Residue): The residue containing the reference atom.
        atom_name (str): Name of the missing atom to estimate.
        reference_atom (str): Name of the reference atom for position estimation.
        bond_length (float): Approximate bond length between reference atom and missing atom.
        direction (np.ndarray): Optional direction vector. If None, use a default vector.
    
    Returns:
        np.ndarray: Estimated coordinates of the missing atom.
    """
    if reference_atom in residue:
        ref_atom_coord = residue[reference_atom].coord
        if direction is None:
            # Use a default direction vector if not provided
            direction = np.array([1.0, 0.0, 0.0])
        
        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)
        estimated_coord = ref_atom_coord + bond_length * direction
        return estimated_coord
    else:
        raise ValueError(f"Reference atom {reference_atom} not found in residue.")
    
def concatenate_structure_arrays(arrays, chain_info,plddt_atoms_indicates=None):
    """
    Concatenate arrays from multiple chains into a single array.
    
    Args:
        arrays: List of numpy arrays from each chain
        chain_info: List of dictionaries containing chain information
        
    Returns:
        numpy.ndarray: Single concatenated array
    """
    # First, process protein/nucleic chains
    protein_arrays = []
    small_molecule_arrays = []
    
    for array, info in zip(arrays, chain_info):
        if info['type'] in ['protein', 'nucleic']:
            protein_arrays.append(array)
            print(info['type'],array.shape)
        else:  # small molecule
            small_molecule_arrays.append(array)
    
    # Concatenate protein/nucleic chains along sequence dimension (dim 1)
    if protein_arrays:
        main_array = np.concatenate(protein_arrays, axis=1)
    else:
        main_array = np.zeros((1, 0, 24, 3))  # Empty protein array
    
    # Process small molecules if any exist
    if small_molecule_arrays:
        # Find maximum number of atoms in any small molecule
        max_atoms = max(arr.shape[1] for arr in small_molecule_arrays)
        
        # Pad each small molecule array to max_atoms
        padded_small_molecules = []
        for arr in small_molecule_arrays:
            pad_width = ((0, 0), (0, 0), (0, 0), (0, 0))
            padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
            padded_small_molecules.append(padded_arr)
        
        # Concatenate small molecules together
        if padded_small_molecules:
            small_mol_array = np.concatenate(padded_small_molecules, axis=1)
            # centre all small molecular atoms

            if plddt_atoms_indicates:
                print(plddt_atoms_indicates)
                print(small_mol_array.shape)
                retained_atoms = small_mol_array[:, plddt_atoms_indicates, :, :]
                center_of_retained_atoms = np.mean(retained_atoms, axis=1, keepdims=True)  # shpe (1, 1, 1, 3)

                for i in range(small_mol_array.shape[1]):
                    if i not in plddt_atoms_indicates:
                        small_mol_array[:, i, :, :] = center_of_retained_atoms

                print("Updated small_mol_array:")
            #def random_rotation_matrix():
            #    
            #    q = np.random.randn(4)
            #    q /= np.linalg.norm(q)
            #    
            #    q0, q1, q2, q3 = q
            #    
            #    R = np.array([
            #        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
            #        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
            #        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
            #    ])
            #    return R

            # 
            #atom_nums=small_mol_array.shape[1]
            # 
            #result = np.empty((8, atom_nums, 1, 3))
            # 
            #for i in range(8):
            #    
            #    R = random_rotation_matrix()
            #    rotated_matrix = np.dot(small_mol_array[0, :, 0, :], R.T)
            #   
            #    result[i, :, 0, :] = rotated_matrix
            #small_mol_array = result
            #main_array = np.tile(main_array, (8, 1, 1, 1))
            # Now pad the small molecule array to match protein array's atom dimension
            if small_mol_array.shape[2] < 24:
                pad_width = ((0, 0), (0, 0), (0, 24 - small_mol_array.shape[2]), (0, 0))
                small_mol_array = np.pad(small_mol_array, pad_width, mode='constant', constant_values=0)
            
            # Concatenate with main array
            main_array = np.concatenate([main_array, small_mol_array], axis=1)
    
    return main_array

def parse_cif_pdb_to_array_with_estimation(file_path, dense_atom_mapping,plddt_atoms_indicates=None):
    """
    Process structure file and return a single concatenated array.
    """
    parser = MMCIFParser(QUIET=True) if file_path.endswith('.cif') else PDBParser(QUIET=True)
    structure = parser.get_structure("molecule", file_path)
    
    arrays = []
    chain_info = []
    
    # Sort chains by chain ID
    chains = sorted(structure.get_chains(), key=lambda x: x.id)
    
    for chain in chains:
        array, chain_type = process_chain(chain, dense_atom_mapping)
        if array is not None:
            arrays.append(array)
            chain_info.append({
                'chain_id': chain.id,
                'type': chain_type
            })
            print(chain.id,chain_type)
    
    if not arrays:
        raise ValueError("No valid chains found in structure")
    
    # Concatenate all arrays into one
    final_array = concatenate_structure_arrays(arrays, chain_info,plddt_atoms_indicates)
    
    return final_array

def setup_logging(output_dir):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
import os
import logging
import pickle
import argparse
from datetime import datetime
from collections import Counter
import concurrent.futures
from threading import Lock

class ThreadSafeCounter:
    def __init__(self):
        self.counter = Counter()
        self.lock = Lock()
    
    def increment(self, key):
        with self.lock:
            self.counter[key] += 1
    
    def get_count(self):
        with self.lock:
            return dict(self.counter)

def process_single_file(args,insert,plddt_atoms_indicates=None):
    """Process a single PDB/CIF file with error handling"""
    pdb_file, input_dir, output_dir = args
    try:
        pdb_path = os.path.join(input_dir, pdb_file)
        out_pkl_path = os.path.join(output_dir, f"{os.path.splitext(pdb_file)[0]}.pkl")
        
        
        # Process based on file extension
        if pdb_file.endswith(".cif") or pdb_file.endswith(".pdb"):
            array = parse_cif_pdb_to_array_with_estimation(pdb_path, DENSE_ATOM,plddt_atoms_indicates)
        logging.info(f"Processing: {pdb_file} {array.shape}")
        # Pad array
        if insert:
            # for ptm
            print(f"before ptm insert{array.shape}")
            index = insert[0] - 1
            num_samples = insert[1] - 1
            target_data = array[:, index - 1, 0, :] 
            mean_values = np.mean(target_data, axis=0)  
            gaussian_noise = np.random.normal(loc=mean_values, scale=0.1, size=(array.shape[0], num_samples, 3))
            gaussian_noise_padded = np.zeros((array.shape[0], num_samples, 24, 3))  

            gaussian_noise_padded[:, :, 0, :] = gaussian_noise 
            array_left = array[:, :index, :, :]  
            array_right = array[:, index:, :, :]  
            array = np.concatenate([array_left, gaussian_noise_padded, array_right], axis=1)
            print(f"after ptm insert{array.shape}")

        padded_array, target_bucket = pad_array_to_bucket(array, buckets=_BUCKETS)
        print(f"Target bucket for {pdb_file}: {target_bucket}")
        
        # Save processed file
        with open(out_pkl_path, 'wb') as f:
            pickle.dump(padded_array, f)
        
        return True, pdb_file, None
        
    except Exception as e:
        return False, pdb_file, str(e)

def process_structure_files(args):
    """Process PDB/CIF files using multiple threads"""
    setup_logging(args.output_dir)
    
    # Initialize thread-safe counters
    stats = ThreadSafeCounter()
    failed_files = []
    
    # Get list of files to process
    try:
        pdb_files = [f for f in os.listdir(args.input_dir)
                     if f.endswith((".pdb", ".cif"))]
    except Exception as e:
        logging.error(f"Error accessing input directory: {str(e)}")
        return None, None
    
    total_files = len(pdb_files)
    if total_files == 0:
        logging.warning(f"No .pdb or .cif files found in {args.input_dir}")
        return stats.get_count(), failed_files
    
    logging.info(f"Found {total_files} files to process")
    start_time = datetime.now()
    
    # Prepare arguments for thread pool
    thread_args = [(pdb_file, args.input_dir, args.output_dir) for pdb_file in pdb_files]
    
    # Calculate optimal number of threads (max 32)
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    processed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_file, arg): arg[0] 
                         for arg in thread_args}
        
        for future in concurrent.futures.as_completed(future_to_file):
            processed_count += 1
            success, pdb_file, error = future.result()
            
            if success:
                stats.increment('successful')
            else:
                stats.increment('failed')
                failed_files.append((pdb_file, error))
                logging.error(f"Failed to process {pdb_file}: {error}")
            
            # Log progress at intervals
            if processed_count % 10 == 0 or processed_count/total_files in {0.25, 0.5, 0.75}:
                progress = (processed_count / total_files) * 100
                logging.info(f"Progress: {progress:.1f}% ({processed_count}/{total_files} files)")
    
    # Calculate processing time
    processing_time = datetime.now() - start_time
    
    # Log summary statistics
    final_stats = stats.get_count()
    logging.info("\nProcessing Summary:")
    logging.info(f"Total files found: {total_files}")
    logging.info(f"Successfully processed: {final_stats.get('successful', 0)}")
    logging.info(f"Failed: {final_stats.get('failed', 0)}")
    logging.info(f"Total processing time: {processing_time}")
    logging.info(f"Used threads: {max_workers}")
    
    # Log failed files if any
    if failed_files:
        logging.info("\nFailed files:")
        for file, error in failed_files:
            logging.info(f"- {file}: {error}")
    
    return final_stats, failed_files

def main():
    parser = argparse.ArgumentParser(description="Process PDB and CIF files to pickled arrays")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing PDB/CIF files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for processed files"
    )
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process files
    stats, failed_files = process_structure_files(args)
    
    # Return appropriate exit code
    return 0 if stats and stats.get('failed', 0) == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)