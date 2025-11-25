
import torch

import os

import logging

import pandas as pd

import numpy as np

import argparse

import sys

import time

from Bio.PDB import MMCIFParser, PDBParser, PDBIO, NeighborSearch
import pyrosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta import MoveMap
pyrosetta.init('-ignore_unrecognized_res -ignore_zero_occupancy -mute all -corrections::beta_nov16 true')
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

import tempfile

logging.basicConfig(level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ])

class InterfaceAnalyzer:
    def __init__(self, params_file=None,
                 ddg_chains_1=('A'),
                ddg_chains_2=('B')):
        self.params_file = params_file
        self.protein_chains = ddg_chains_1
        self.ligand_chains = ddg_chains_2
        self._init_pyrosetta()
        self.analyzer = InterfaceAnalyzerMover()
        self._configure_analyzer()

        self._configure_relax()

    def _init_pyrosetta(self):
        """Initialize PyRosetta and load parameter files."""
        extra_options = "-mute all"
        if self.params_file:
            extra_options += f" -extra_res_fa {self.params_file}"
        pyrosetta.init(extra_options=extra_options)

    def _configure_analyzer(self):
        """Configure analyzer options."""
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
        """Configure the Relax protocol."""
        self.relax_protocol = FastRelax()
        self.relax_protocol.set_scorefxn(self.scorefxn)
        
        mm = MoveMap()
        mm.set_bb(True)
        mm.set_chi(True)
        self.relax_protocol.set_movemap(mm)

    def analyze_interface(self, pdb_path,out_path):
        try:
            self._init_pyrosetta()

            pose = self._load_and_validate_pdb(pdb_path, self.protein_chains, self.ligand_chains)
            def check_residue_atoms(pose, res_id):
                res = pose.residue(res_id)
                print(f"Residue {res_id} ({res.name()}) Atoms:")
                for i in range(1, res.natoms() + 1):
                    print(f"  Atom {i}: {res.atom_name(i).strip()}")
            
            check_residue_atoms(pose, pose.total_residue())  # Check the last residue (small molecule)
            # Set up MoveMap (explicitly disable the ligand)
            #mm = pyrosetta.MoveMap()
            #mm.set_bb(False)
            #mm.set_chi(False)
            #
            #protein_chains_set = set(self.protein_chains)
            #ligand_chains_set = set(self.ligand_chains)
            #for res in range(1, pose.total_residue() + 1):
            #    chain = pose.pdb_info().chain(res)
            #    if chain in protein_chains_set:
            #        mm.set_bb(res, True)
            #        mm.set_chi(res, True)
            #    elif chain in ligand_chains_set: 
#
            #        mm.set_bb(res, False)
            #        mm.set_chi(res, False)
            print("Applying Relax protocol...")
            self.relax_protocol.apply(pose)
            
            protein_chain_str = ''.join(self.protein_chains)
            ligand_chain_str = ''.join(self.ligand_chains)
            self.analyzer.set_interface(f"{protein_chain_str}_{ligand_chain_str}")
            self.analyzer.apply(pose)
            if out_path.endswith(".cif"):
                out_path = out_path.replace(".cif",".pdb")
            pose.dump_pdb(out_path)
            return self._compile_results()
        except Exception as e:
            print(f"Analysis failed: {e}")
            return None
        
    def _load_and_validate_pdb(self, pdb_path, protein_chains, ligand_chains):
        """Load and validate PDB structure."""
        if pdb_path.endswith(".pdb"):
            pose = pyrosetta.pose_from_pdb(pdb_path)
        else:
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as temp_pdb_file:
                convert_cif_to_pdb(pdb_path, temp_pdb_file.name)
                pose = pyrosetta.pose_from_pdb(temp_pdb_file.name)
                os.remove(temp_pdb_file.name)

        chains = {pose.pdb_info().chain(res) for res in range(1, pose.total_residue()+1)}
        for chain_group in [protein_chains, ligand_chains]:
            for chain in chain_group:
                if chain not in chains:
                    raise ValueError(f"Chain {chain} not found in the PDB")

        for chain in ligand_chains:
            lig_residues = [res for res in range(1, pose.total_residue()+1) 
                            if pose.pdb_info().chain(res) == chain]
            if not lig_residues:
                raise ValueError(f"No residues found in ligand chain {chain}")
        
        print(f"Successfully loaded: {pdb_path}, Total residues: {pose.total_residue()}")
        return pose

    def _compile_results(self):
        """Compile result data."""
        data = self.analyzer.get_all_data()
        return {
            'interface_sc': data.sc_value,
            'interface_interface_hbonds': data.interface_hbonds,
            'interface_dG': self.analyzer.get_interface_dG(),
            'interface_dSASA': data.dSASA,
            'interface_packstat': data.packstat,
            'interface_dG_SASA_ratio': data.dG_dSASA_ratio,
        }
        



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

                    #if resname not in amino_acids:  
                    #    residue.resname = "LIG"  # modify resname for further process 
        
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_file)

        print(f"convert successfully, save as {pdb_file}")
        return True
    except Exception as e:
        print(f"convert failed, {str(e)}")
        return False
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-metadata", type=str, default=None, help="csv file contains the information of scaffold")
    parser.add_argument("-shuffle", action="store_true", help='Whether to shuffle the metadata before running')
    parser.add_argument("-start_idx", type=int, default=0, help="The start index of metadata to run")
    parser.add_argument("-end_idx", type=int, default=None, help="The end index of metadata to run (default: None, run till end)")
    parser.add_argument('-ddg_target_chain_1', type=str, help="chains to calculate ddg, e.g. 'A,B'", default='')
    parser.add_argument('-ddg_target_chain_2', type=str, help="chains to calculate ddg, e.g. 'C,D'", default='')
    parser.add_argument("-interface_analyzer", action="store_true")
    parser.add_argument("-output_dir", type=str, default='./', help='The directory of prediction result')
    parser.add_argument("-suffix", type=str, default=None, help='suffix to append to the output files')
    parser.add_argument('-extra_res_params', type=str, default=None, help='extra res params location, must input in ddg with small molecular')
    args = parser.parse_args(sys.argv[1:])


    ddg_target_chain_1 = tuple(args.ddg_target_chain_1.split(','))
    ddg_target_chain_2 = tuple(args.ddg_target_chain_2.split(','))
    analyzer = InterfaceAnalyzer(args.extra_res_params, ddg_target_chain_1, ddg_target_chain_2)

    os.makedirs(args.output_dir, exist_ok=True)
    relax_out= os.path.join(args.output_dir,"relax")
    os.makedirs(relax_out, exist_ok=True)
    metadata_df = pd.read_csv(args.metadata)
    metadata_df = metadata_df.dropna(subset=['eval_path'])
    if args.shuffle:
        metadata_df = metadata_df.sample(frac=1, random_state=42).reset_index(drop=True)
    if args.end_idx is None or args.end_idx > len(metadata_df):
        args.end_idx = len(metadata_df)
    metadata_df = metadata_df.iloc[args.start_idx:args.end_idx]
    metadata_df = metadata_df.sort_values('length', ascending=True) if 'length' in metadata_df.columns else metadata_df

    print(f"Processing metadata rows from {args.start_idx} to {args.end_idx} (total {len(metadata_df)} rows)")

    for scaffold_path, scaffold_group in metadata_df.groupby('eval_path'):
        start_time = time.time()
        scaffold_group = scaffold_group.reset_index(drop=True)
        row = scaffold_group.iloc[0]
        self_consistency_results = []
        print(scaffold_group)

        result = {}
        out_path = os.path.join(relax_out,os.path.basename(scaffold_path))
        interface_results = analyzer.analyze_interface(scaffold_path,out_path)
        result.update(interface_results)
        self_consistency_results.append(result)

        predict_df = pd.DataFrame(self_consistency_results)

        # Merge predict_df with metadata_df

        merged_df = pd.merge(scaffold_group, predict_df, left_index=True, right_index=True)

        logging.info(f'Run rosetta evaluation for scaffold {row["packed_path"]} with {len(merged_df)} seqs in {time.time()-start_time:.2f}s')
        logging.info(merged_df)

        metadata_path = os.path.join(args.output_dir, "ddg_prediction.csv")

        merged_df.to_csv(metadata_path, mode='a', header=not os.path.exists(metadata_path), index=False)
        
