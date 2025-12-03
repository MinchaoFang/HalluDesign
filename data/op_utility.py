from Bio.PDB import MMCIFParser, PDBParser, PDBIO
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import numpy as np

from Bio.PDB import PDBParser, PDBIO, Residue, Atom

import random
from Bio.PDB import Structure, Model, Chain, Residue
import numpy as np
def read_protein_info_from_copied_file(copied_file):
    parser = PDBParser()
    structure = parser.get_structure("protein", copied_file)
    
    protein_info = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id

            if chain_id not in protein_info:
                protein_info[chain_id] = []
            for residue in chain:

                protein_info[chain_id].append(residue)
    return protein_info

from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder, three_to_one
import warnings
import os

# Suppress warnings that Biopython might issue during parsing
warnings.filterwarnings('ignore', category=UserWarning, module='Bio.PDB')

def find_all_sequence_residues_in_pdb(pdb_file_path: str, query_sequences: list[str]) -> dict[str, list[str]]:
    """
    Search for all occurrences of multiple query protein sequences within a PDB or CIF file, 
    and return the chain IDs and residue indices for each matched segment.

    Args:
        pdb_file_path (str): Full path to the PDB or CIF file.
        query_sequences (list[str]): A list of query protein sequences to search for 
                                     (e.g., ["ABC", "XYZ"]). Each sequence should be 
                                     represented using single-letter amino acid codes 
                                     (e.g., 'A', 'K', 'S').

    Returns:
        dict[str, list[str]]: A dictionary mapping each query sequence to a list containing 
                              all residue identifiers (chain ID + residue index) for every 
                              matching instance found in the structure.
                              
                              Example:
                                  {
                                      "AK": ["A1", "A2"],
                                      "ST": ["A3", "A4"],
                                      "GV": ["B1", "B2"],
                                      "KST": ["A2", "A3", "A4"]
                                  }

                              If a query sequence is not found, it will not appear 
                              as a key in the returned dictionary.
    """

    all_results = {query_seq: [] for query_seq in query_sequences} # Initialize results for all queries

    if not os.path.exists(pdb_file_path):
        print(f"error: no PDB/CIF file '{pdb_file_path}'")
        return {}
    
    if pdb_file_path.lower().endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    elif pdb_file_path.lower().endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        print(f"error '{pdb_file_path}'ecpect .pdb or .cif")
        return {}

    try:
        structure = parser.get_structure("protein", pdb_file_path)
    except Exception as e:
        print(f"error '{pdb_file_path}': {e}")
        return {}

    ppb = PPBuilder()

    chain_data = {} 

    for model in structure:
        model_id = model.id
        for chain in model: 
            polypeptides = ppb.build_peptides(chain) 
            chain_id = chain.id 

            for pp in polypeptides:
                chain_sequence_str = ""
                residue_details_map = [] 

                for residue in pp:
                    res_name = residue.get_resname()
                    res_id_tuple = residue.get_id() 
                    
                    try:
                        one_letter_code = three_to_one(res_name)
                        chain_sequence_str += one_letter_code
                        residue_details_map.append((res_id_tuple[1], res_id_tuple[2]))
                    except KeyError:
                        pass
                
                if chain_sequence_str:
                    chain_data[(model_id, chain_id)] = (chain_sequence_str, residue_details_map)
    
    # Iterate through each query sequence
    for query_seq in query_sequences:
        # Iterate through all chains we extracted data from
        for (model_id, chain_id), (chain_seq_str, res_map) in chain_data.items():
            
            # Find all occurrences of the query_seq in the current chain's sequence
            current_index = -1
            while True:
                current_index = chain_seq_str.find(query_seq, current_index + 1)
                if current_index == -1:
                    break # No more occurrences found

                # If a match is found, extract all residue indices for this match
                match_residues = []
                for i in range(len(query_seq)):
                    res_index_in_map = current_index + i
                    if res_index_in_map < len(res_map): # Ensure we don't go out of bounds
                        found_res_seq_id = res_map[res_index_in_map][0]
                        found_res_ins_code = res_map[res_index_in_map][1]

                        formatted_res = f"{chain_id}{found_res_seq_id}"
                        if found_res_ins_code and found_res_ins_code.strip() != '':
                            formatted_res += found_res_ins_code
                        match_residues.append(formatted_res)
                
                # Add the list of residues for this match to the results for the current query sequence
                # We extend the list as there might be multiple matches for the same query sequence
                all_results[query_seq].extend(match_residues)
                
    # Remove duplicates within the list for each query if needed (e.g., if a sequence like "AAA" matches "AA" multiple times overlapping)
    # This example ensures unique entries for each specific residue, but if you want to keep
    # distinct 'matches' even if they share some residues, you might need a more complex structure.
    all = []
    for query_seq in all_results:
        all += sorted(list(set(all_results[query_seq]))) # Sort for consistent output

    return all

def filter_protein_info(protein_info, fixed_chains, fixed_residues, redesign_res):
    """
    Filter protein structure information with two supported modes:
    
    1. Full-chain fixation — retain entire chains as-is.
    2. Cross-chain residue fixation — retain only specified residues from other chains.

    Args:
        protein_info (dict): Dictionary representing the protein structure, 
                             in the format {chain_id: [residue objects]}.
        fixed_chains (list): List of chain IDs to be fully retained.
        fixed_residues (list): List of specific residues to retain across chains, 
                               formatted as ["A100", "B50"], where each entry 
                               combines chain ID and residue index.

    Returns:
        dict: A filtered protein structure dictionary containing only the retained chains 
              and/or residues based on the specified fixation mode.
    """

    filtered = {}

    for chain_id in protein_info:
        if chain_id in fixed_chains:
            filtered[chain_id] = protein_info[chain_id]

        else:
            #if fixed_residues:
            #    fixed_residues_set = set(fixed_residues.split(" "))
            #else:
            #    fixed_residues_set = set()
            filtered_res = [
                res for res in protein_info[chain_id]
                if f"{chain_id}{residue_to_str(res.id)}" in fixed_residues
            ]
            if filtered_res:
                filtered[chain_id] = filtered_res
    if redesign_res:
        for chain_id in protein_info:
            filtered_res = [
                res for res in protein_info[chain_id]
                if f"{chain_id}{residue_to_str(res.id)}" not in redesign_res
            ]
            #print([
            #    res for res in protein_info[chain_id]
            #    if f"{chain_id}{residue_to_str(res.id)}" in redesign_res
            #])
            #print(redesign_res)
            if filtered_res:
                filtered[chain_id] = filtered_res
            
    return filtered

def residue_to_str(res_id):
    """Convert a residue ID into a standardized string format, 
        handling insertion codes and other special cases."""
        # Example:
        # res_id = (' ', 100, 'A') → "100A"

    hetero, resnum, insert = res_id

    return f"{resnum}{insert if insert != ' ' else ''}"


from Bio.PDB import PDBParser, MMCIFParser

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

import numpy as np

def parse_cif(cif_path):
    mmcif_dict = MMCIF2Dict(cif_path)
    
    atom_names = mmcif_dict["_atom_site.label_atom_id"]
    chain_ids = mmcif_dict["_atom_site.auth_asym_id"]
    x_coords = list(map(float, mmcif_dict["_atom_site.Cartn_x"]))
    y_coords = list(map(float, mmcif_dict["_atom_site.Cartn_y"]))
    z_coords = list(map(float, mmcif_dict["_atom_site.Cartn_z"]))
    res_seq_ids = mmcif_dict["_atom_site.label_seq_id"]  # Residue sequence ID

    return atom_names, chain_ids, x_coords, y_coords, z_coords, res_seq_ids

def parse_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_path)
    
    atom_names = []
    chain_ids = []
    x_coords = []
    y_coords = []
    z_coords = []
    res_seq_ids = []

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_names.append(atom.get_name())
                    chain_ids.append(chain.id)
                    x, y, z = atom.coord

                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(z)
                    res_seq_ids.append(residue.get_id()[1])  

    return atom_names, chain_ids, x_coords, y_coords, z_coords, res_seq_ids


from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

def smiles_to_mmcif(smiles, molecule_id='MY-X7F', molecule_name='5,8-bis(oxidanyl)naphthalene-1,4-dione'):
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    # Remove hydrogens
    mol = Chem.RemoveHs(mol)
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    # Calculate formula and formula weight
    formula = rdMolDescriptors.CalcMolFormula(mol)
    formula_weight = rdMolDescriptors.CalcExactMolWt(mol)
    # Generate InChI and InChIKey
    inchi = Chem.MolToInchi(mol)
    inchi_key = Chem.InchiToInchiKey(inchi)
    
    # Prepare mmCIF content
    mmcif_lines = [
        f"data_{molecule_id}",
        "#",
        f"_chem_comp.id {molecule_id}",
        f"_chem_comp.name '{molecule_name}'",
        "_chem_comp.type non-polymer",
        f"_chem_comp.formula '{formula}'",
        "_chem_comp.mon_nstd_parent_comp_id ?",
        "_chem_comp.pdbx_synonyms ?",
        f"_chem_comp.formula_weight {formula_weight:.3f}",
        "#"
    ]
    
    # Atom information section
    mmcif_lines.extend([
        "loop_",
        "_chem_comp_atom.comp_id",
        "_chem_comp_atom.atom_id",
        "_chem_comp_atom.type_symbol",
        "_chem_comp_atom.charge",
        "_chem_comp_atom.pdbx_leaving_atom_flag",
        "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
        "_chem_comp_atom.pdbx_model_Cartn_z_ideal"
    ])
    
    # Atom name mapping
    atom_names = {}
    non_h_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
   
    # Create atom names using RDKit-friendly nomenclature
    for atom in non_h_atoms:
        symbol = atom.GetSymbol()
        # Get the atom's index in the original molecule
        idx = atom.GetIdx()
        
        # Generate atom name based on element and its index
        # Use a format like C1, O2, N3, etc.
        atom_name = f"{symbol}{idx+1}"
        atom_names[idx] = atom_name
    
    # Add atom coordinates and ideal coordinates
    conf = mol.GetConformer()
    for atom in non_h_atoms:
        idx = atom.GetIdx()
        atom_id = atom_names[idx]
        coords = conf.GetAtomPosition(idx)
        mmcif_lines.append(
            f"{molecule_id} {atom_id} {atom.GetSymbol()} 0 N {coords.x:.3f} {coords.y:.3f} {coords.z:.3f}"
        )
    
    # Bond information section
    mmcif_lines.extend([
        "#",
        "loop_",
        "_chem_comp_bond.atom_id_1",
        "_chem_comp_bond.atom_id_2",
        "_chem_comp_bond.value_order",
        "_chem_comp_bond.pdbx_aromatic_flag"
    ])
    
    # Add bond information
    for bond in mol.GetBonds():
        # Exclude bonds involving hydrogen atoms
        if (bond.GetBeginAtom().GetAtomicNum() != 1 and
            bond.GetEndAtom().GetAtomicNum() != 1):
            atom1_id = atom_names[bond.GetBeginAtomIdx()]
            atom2_id = atom_names[bond.GetEndAtomIdx()]
           
            # Determine bond order
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond_order = "SING"
            elif bond.GetBondType() == Chem.BondType.DOUBLE:
                bond_order = "DOUB"
            elif bond.GetBondType() == Chem.BondType.TRIPLE:
                bond_order = "TRIP"
            else:
                bond_order = "SING"
           
            # Check if bond is aromatic
            aromatic_flag = "Y" if bond.GetIsAromatic() else "N"
           
            mmcif_lines.append(
                f"{atom1_id} {atom2_id} {bond_order} {aromatic_flag}"
            )
    
    # Descriptor information section
    mmcif_lines.extend([
        "#",
        "loop_",
        "_pdbx_chem_comp_descriptor.comp_id",
        "_pdbx_chem_comp_descriptor.type",
        "_pdbx_chem_comp_descriptor.program",
        "_pdbx_chem_comp_descriptor.program_version",
        "_pdbx_chem_comp_descriptor.descriptor",
        f"{molecule_id} SMILES RDKit 2024 {smiles}",
        f"{molecule_id} InChI RDKit 2024 {inchi}",
        f"{molecule_id} InChIKey RDKit 2024 {inchi_key}",
        "#"
    ])
    
    return '\n'.join(mmcif_lines)


def get_residue_centroid(residue):
    atoms = list(residue.get_atoms())
    coords = np.array([atom.get_coord() for atom in atoms])
    centroid = np.mean(coords, axis=0)
    return centroid


def generate_gaussian_noise(mean, std, size=3):
    return np.random.normal(loc=mean, scale=std, size=size)


def create_glycine_residue(residue_id, centroid, chain_id):
    residue = Residue.Residue((' ', residue_id, ' '), 'GLY', ' ')
    ca = Atom.Atom('CA', centroid + generate_gaussian_noise(0, 0.5), 0.0, 1.0, ' ', 'CA', 1, 'C')
    n = Atom.Atom('N', centroid + generate_gaussian_noise(0, 0.5), 0.0, 1.0, ' ', 'N', 1, 'N')
    c = Atom.Atom('C', centroid + generate_gaussian_noise(0, 0.5), 0.0, 1.0, ' ', 'C', 1, 'C')
    o = Atom.Atom('O', centroid + generate_gaussian_noise(0, 0.5), 0.0, 1.0, ' ', 'O', 1, 'O')
    ca.segid = chain_id

    n.segid = chain_id

    c.segid = chain_id

    o.segid = chain_id

    residue.add(ca)
    residue.add(n)
    residue.add(c)
    residue.add(o)
    return residue


def parse_input(input_string):
    ranges = input_string.split()
    parsed_ranges = []
    for r in ranges:
        chain_id = r[0]  

        start, end = map(int, r[1:].split('-'))

        parsed_ranges.append((chain_id, start, end))
    return parsed_ranges

def modify_range(start, end, cdr_2, min_length=5, max_extend=5):
    if cdr_2:
        operation = "same"
    else:
        operation = random.choice(["extend", "same"]) #  "shrink",
    if operation == "extend":
        new_start = start 

        new_end = end + random.randint(1, max_extend)
    elif operation == "shrink":
        if end - start + 1 > min_length:
            new_start = start 
            new_end = max(end - random.randint(1, max_extend), new_start + min_length - 1)
        else:
            new_start, new_end = start, end

    else:
        new_start, new_end = start, end

    return new_start, new_end

def reindex_atoms(structure):
    """
    Reassign correct atom indices (serial numbers) for all atoms 
    in a PDB structure, and set the segment ID of residues 
    (segID) to their corresponding chain ID if missing.

    Args:
        structure (Bio.PDB.Structure.Structure): The structure object to process.
    """

    atom_index = 1  

    for model in structure:
        for chain in model:
            for residue in chain:
                # if Residue segid missing ID

                if not residue.segid or residue.segid.strip() == "":
                    residue.segid = chain.id

                for atom in residue:
                    # rearrange serial_number

                    atom.serial_number = atom_index

                    atom_index += 1

def reindex_structure(original_structure, new_start_res_id=1):
    """
    Create a new structure with residues renumbered for all chains.

    Args:
        original_structure (Bio.PDB.Structure): The original structure object.
        new_start_res_id (int, optional): Starting residue number. Defaults to 1.

    Returns:
        Bio.PDB.Structure: A new structure object with renumbered residues.
    """

    new_structure = Structure.Structure(original_structure.id)  # new Structure

    new_structure.level = "S"  # set different level

    
    for model in original_structure:
        new_model = Model.Model(model.id)  # new Model

        
        for chain in model:
            new_chain = Chain.Chain(chain.id)  # new Chain

            new_res_id = new_start_res_id  # reindex

            
            for residue in chain.get_residues():
                # new Residue，copy old Residue

                new_residue = Residue.Residue(
                    id=(' ', new_res_id, ' '),

                    resname=residue.resname,
                    segid=residue.segid

                )
                
                for atom in residue.get_atoms():
                    new_residue.add(atom)
                
                # Add Residue to new Chain

                new_chain.add(new_residue)
                new_res_id += 1  

                
            # Add Chain to Model

            new_model.add(new_chain)
        
        # Add Model to new Structure

        new_structure.add(new_model)
    
    return new_structure

def insert_residues(chain, new_residues, insert_start):
    """
    Insert new residues into a chain at the specified position 
    and return a new chain object.

    Args:
        chain (Bio.PDB.Chain): The original chain object.
        new_residues (list): A list of new residues to insert.
        insert_start (int): The starting position (residue index) for insertion.

    Returns:
        Bio.PDB.Chain: A new chain object containing the inserted residues.
    """

    new_chain = Chain.Chain(chain.id)  # new chain
    
    for residue in chain.get_residues():
        if residue.id[1] < insert_start:
            new_chain.add(residue)
        elif residue.id[1] > insert_start:
            if (residue.id[1]-1000) == insert_start:
                for new_residue in new_residues:
                    new_chain.add(new_residue)
            new_chain.add(residue)
    return new_chain

def cdr_process(input_string, pdb_file, output_file):
    parser = PDBParser(QUIET=True)
    io = PDBIO()
    structure = parser.get_structure("structure", pdb_file)
    parsed_ranges = parse_input(input_string)
    chain_number_list = []
    res_modify_count = {"A":0,"B":0,"C":0}
    seg_count = 0
    for chain_id, start, end in parsed_ranges:
        print(f"Processing chain {chain_id}, range {start}-{end}")
        seg_count += 1
        # find centre of all atom
        start = res_modify_count[chain_id] + start
        end = res_modify_count[chain_id]+ end
        print(f"Processing chain {chain_id}, range {start}-{end}")
        centroids = []
        residues_to_delete = []
        for res_id in range(start, end + 1):
            try:
                residue = structure[0][chain_id][res_id]
                centroid = get_residue_centroid(residue)
                centroids.append(centroid)
                residues_to_delete.append(res_id) 

            except KeyError:
                print(f"Residue {chain_id}:{res_id} not found, skipping.")

        if not centroids:
            print(f"No valid residues found in chain {chain_id}, range {start}-{end}. Skipping.")
            continue

        # centre 

        centroid_avg = np.mean(centroids, axis=0)

        # delete all residues

        for res_id in residues_to_delete:
            try:
                structure[0][chain_id].detach_child((' ', res_id, ' '))
            except KeyError:
                continue

        # random expand or delete
        if seg_count % 6 == 2:
            print("cdr_2")
            cdr_2 = True
        else:
            cdr_2 = False
        new_start, new_end = modify_range(start, end,cdr_2)
        print(f"Modified range for chain {chain_id}: {new_start}-{new_end}")
        #new_start = res_modify_count[chain_id] + new_start
        #new_end = res_modify_count[chain_id]+ new_end
        #print(f"Modified range for chain {chain_id}: {new_start}-{new_end}")
        print("res mod",new_end-end)
        res_modify_count[chain_id] = res_modify_count[chain_id] + new_end-end

        # modify other residues

        insert_count = new_end - new_start + 1

        sorted_residues = sorted(
            [residue for residue in structure[0][chain_id].get_residues()],
            key=lambda residue: residue.id[1],
            reverse=True,  # ranking

        )

        for residue in sorted_residues:
            if residue.id[1] > residues_to_delete[-1]: 

                residue.id = (' ', residue.id[1] + 1000 - len(residues_to_delete), ' ')

        # insert new glys

        new_residues = []
        for i, res_id in enumerate(range(new_start, new_end + 1)):
            new_residue = create_glycine_residue(res_id, centroid_avg, chain_id)
            new_residues.append(new_residue)
            chain_number_list.append(f"{chain_id}{res_id}")

        chain = structure[0][chain_id]
        new_chain = insert_residues(chain, new_residues, new_start)
        structure[0].detach_child(chain.id)  

        structure[0].add(new_chain)  

        structure=reindex_structure(structure, new_start_res_id=1)
        # reindex before save

        reindex_atoms(structure)
        #io.set_structure(structure)
        #io.save(output_file.replace(".pdb", f"_{chain_id}_{start}.pdb"))

    # save file

    io.set_structure(structure)
    io.save(output_file)
    print(f"Saved modified PDB file to {output_file}")
    return chain_number_list
            
import random

def random_protein_sequence(sequence, keep_positions, chain):
    """
    Args:
        sequence (str): Original amino acid sequence.
        keep_positions (list[str]): List of positions to keep unchanged 
                                    (e.g., ["A12", "A55"]).
        chain (str): Chain ID (e.g., "A").

    Returns:
        dict: A dictionary containing the mutated sequence in the format 
              {chain: mutated_seq}.
    """

    print(sequence, keep_positions, chain)
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    keep_positions = set(keep_positions)  
    mutated = []

    for i, aa in enumerate(sequence, start=1):  
        pos_str = f"{chain}{i}"
        if pos_str in keep_positions:
            mutated.append(aa) 
        else:
            choices = [x for x in amino_acids]  
            mutated.append(random.choice(choices))  

    return  "".join(mutated)

import random
from Bio.PDB import PDBParser

def get_random_close_residues(pdb_file, chain_a="A", chain_b="B", cutoff=6.0):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_file)

    model = structure[0]
    if chain_a not in model or chain_b not in model:
        raise ValueError(f"PDB no {chain_a} or {chain_b}")

    chainA = model[chain_a]
    chainB = model[chain_b]

    # collect residue  < cutoff 
    close_pairs = []
    for resA in chainA:
        if "CA" not in resA:  
            continue
        caA = resA["CA"].get_coord()
        for resB in chainB:
            if "CA" not in resB:
                continue
            caB = resB["CA"].get_coord()
            dist = ((caA - caB) ** 2).sum() ** 0.5
            if dist < cutoff:
                close_pairs.append((resA.get_id()[1], resB.get_id()[1]))

    if not close_pairs:
        return None

    return random.choice(close_pairs)


from Bio.PDB import PDBParser

import numpy as np

def residue_centroid(residue):
    """find centre of residue"""
    coords = np.array([atom.get_coord() for atom in residue.get_atoms()])
    return coords.mean(axis=0)

def chain_centroid(chain):
    """find centre of chain"""
    coords = np.array([atom.get_coord() for atom in chain.get_atoms()])
    return coords.mean(axis=0)

def most_central_residue_resseq(pdb_file, chain_id='A'):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_file)


    model = structure[0]
    if chain_id not in model:
        raise ValueError(f"chain {chain_id} missing")
    
    chain = model[chain_id]

    chain_ctr = chain_centroid(chain)
    

    min_dist = 1e9

    central_resseq = None

    
    for residue in chain.get_residues():
        # ignore water

        if residue.id[0] != ' ':
            continue

        
        res_ctr = residue_centroid(residue)
        dist = np.linalg.norm(res_ctr - chain_ctr)
        
        if dist < min_dist:
            min_dist = dist

            central_resseq = residue.id[1]  # resseq

    
    return central_resseq
