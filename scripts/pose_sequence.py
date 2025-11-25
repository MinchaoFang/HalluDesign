import pyrosetta
from pyrosetta import *
from pyrosetta.rosetta.core.pose import *
from pyrosetta.rosetta.protocols.comparative_modeling import *
from pyrosetta.rosetta.protocols.simple_moves import *

def initialize_pyrosetta():
    """Initialize PyRosetta with recommended options."""
    init_options = "-ex1 -ex2 -use_input_sc -ignore_unrecognized_res -ignore_waters -mute all"
    pyrosetta.init(init_options)

def parse_sequences(sequence_string):
    """Parse multi-chain sequence string separated by ':'."""
    return sequence_string.split(':')

def get_chain_bounds(pose, chain_id='A'):
    """Get the start and end residue indices of a given chain."""
    start_res = 1
    end_res = 0
    
    for i in range(1, pose.total_residue() + 1):
        if pose.pdb_info().chain(i) == chain_id:
            if end_res == 0:
                start_res = i
            end_res = i
    
    if end_res == 0:
        raise ValueError(f"Chain {chain_id} not found in the structure.")
        
    return start_res, end_res

def mutate_chain_sequence(pose, new_sequence, start_res, end_res):
    """
    Replace the amino acid sequence of a specified chain region.

    Parameters:
        pose: PyRosetta Pose object.
        new_sequence (str): The new amino acid sequence.
        start_res (int): Starting residue index.
        end_res (int): Ending residue index.
    """
    mutater = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
    
    for i, aa in enumerate(new_sequence):
        pose_position = start_res + i
        if pose_position > end_res:
            break
            
        # Convert one-letter amino acid code to PyRosetta residue name
        aa_name = pyrosetta.rosetta.core.chemical.aa_from_oneletter_code(aa)
        
        # Apply mutation
        mutater.set_res_name(aa_name)
        mutater.set_target(pose_position)
        mutater.apply(pose)

def build_full_atom_model(pdb_file, sequence_string):
    """
    Build a full-atom protein model from a PDB file and target sequence.
    Non-protein residues (e.g., ligands) are retained but not modified.
    """
    # Initialize PyRosetta
    initialize_pyrosetta()
    
    # Parse the sequence
    chains = parse_sequences(sequence_string)
    chain_a_seq = chains[0]
    
    # Load the input PDB structure
    pose = pyrosetta.pose_from_pdb(pdb_file)
    
    # Get Chain A residue range
    chain_a_start, chain_a_end = get_chain_bounds(pose, 'A')
    chain_a_length = chain_a_end - chain_a_start + 1
    
    # Validate sequence length
    if chain_a_length != len(chain_a_seq):
        raise ValueError(f"Length mismatch for Chain A: sequence ({len(chain_a_seq)}) "
                         f"vs. structure ({chain_a_length})")
    
    # Replace Chain A sequence
    mutate_chain_sequence(pose, chain_a_seq, chain_a_start, chain_a_end)
    
    # Rebuild side chains for Chain A
    rebuild_chain_a_sidechains(pose, chain_a_start, chain_a_end)
    
    # Handle small molecules (ligands, ions, etc.)
    for i in range(1, pose.total_residue() + 1):
        residue = pose.residue(i)
        chain_id = pose.pdb_info().chain(i)
        
        # Skip non-protein residues but retain them in the pose
        if not residue.is_protein():
            print(f"Detected non-protein residue: {residue.name()} in chain {chain_id}, retained in place.")
            continue
    
    return pose, chain_a_start, chain_a_end

def rebuild_chain_a_sidechains(pose, start_res, end_res):
    """Repack or rebuild side chains for Chain A residues."""
    task_factory = pyrosetta.rosetta.core.pack.task.TaskFactory()
    
    # Basic setup
    task_factory.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    task_factory.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())
    
    # Prevent repacking for residues outside Chain A
    prevent_repacking = pyrosetta.rosetta.core.pack.task.operation.PreventRepacking()
    for i in range(1, pose.total_residue() + 1):
        if i < start_res or i > end_res:
            prevent_repacking.include_residue(i)
    task_factory.push_back(prevent_repacking)
    
    # Create and apply packer
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover()
    packer.task_factory(task_factory)
    packer.apply(pose)

def optimize_chain_a(pose, start_res, end_res):
    """Perform local energy minimization for Chain A."""
    scorefxn = pyrosetta.get_fa_scorefxn()
    
    # Define which degrees of freedom to minimize
    movemap = pyrosetta.rosetta.core.kinematics.MoveMap()
    for i in range(1, pose.total_residue() + 1):
        if start_res <= i <= end_res:
            movemap.set_chi(i, True)
            movemap.set_bb(i, False)
    
    # Minimize using LBFGS
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        movemap,
        scorefxn,
        'lbfgs_armijo_nonmonotone',
        0.001,
        True,
        False
    )
    
    min_mover.apply(pose)

def process(pdb_file, sequence_string, output_file):
    """Main processing function."""
    try:
        # Build the model
        pose, chain_a_start, chain_a_end = build_full_atom_model(pdb_file, sequence_string)
        
        # Optionally optimize or validate
        # optimize_chain_a(pose, chain_a_start, chain_a_end)
        # validation_results = validate_chain_a(pose, chain_a_start, chain_a_end)
        # print("Validation results:", validation_results)
        
        # Save final model
        pose.dump_pdb(output_file)
        print(f"Model saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


import pandas as pd
import os
import argparse

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Process PDB files with sequences from CSV')
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to the input CSV file')
    parser.add_argument('--output_dir', type=str, default='output_models',
                       help='Directory for output files (default: output_models)')
    parser.add_argument('--begin_row', type=int, default=0,
                       help='Starting row index (default: 0)')
    parser.add_argument('--end_row', type=int, default=None,
                       help='Ending row index (default: process all rows)')
    
    return parser.parse_args()

def process_pdb_files(args):
    """Read a CSV file and process each PDB/sequence pair."""
    df = pd.read_csv(args.csv_path)
    
    # Create output directory if not exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine processing range
    end_row = args.end_row if args.end_row is not None else len(df)
    df_subset = df.iloc[args.begin_row:end_row]
    
    # Iterate through rows
    for index, row in df_subset.iterrows():
        pdb_file = row['scaffold_path']
        sequence_string = row['mpnn_sequence']
        output_file = os.path.join(
            args.output_dir,
            f"{os.path.basename(row['esmfold_path']).replace('esmfold_','').replace('.pdb','')}.pdb"
        )
        
        print(f"Processing row {index + 1}: PDB file = {pdb_file}, sequence = {sequence_string}")
        
        try:
            process(pdb_file, sequence_string, output_file)
        except Exception as e:
            print(f"Error processing row {index + 1}: {e}")

def main():
    """Entry point for command-line execution."""
    args = parse_arguments()
    process_pdb_files(args)

if __name__ == "__main__":
    main()
