import os
import argparse
import pandas as pd
import sys
from pathlib import Path
import json
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
import esm.esmfold.v1.pretrained
import esm.esmfold.v1.esmfold
from copy import deepcopy
import json
import csv
import os
import sys
sys.path.insert(0,"/storage/caolongxingLab/fangminchao/Proteus/Proteus_flow_matching")
from data import residue_constants as rc
from omegaconf import OmegaConf
from colabdesign.af.model import mk_af_model
from ProteinMPNN.protein_mpnn_utils import model_init
from ProteinMPNN.protein_mpnn_pyrosetta import mpnn_design
from data.parsers import from_pdb_string
from scripts.self_consistency_evaluation import run_folding_and_evaluation
from data import protein
from Bio.PDB import MMCIFParser, PDBIO, MMCIFIO, PDBParser
from Bio.PDB import Structure, Model
import torch
import pose_sequence 

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
    parser.add_argument('--template_path', type=str, required=True,
                       help='Path to template.json file')
    parser.add_argument('--reconstruct', type=str, required=True,
                       help='pose_seq_scaffold or af2 or none')
    parser.add_argument('--alphafold', type=bool,  default=True,
                       help='AF2 or esmfold')
    parser.add_argument('--multimer', type=bool,  default=True,
                       help='AF2 multimer or not')
    parser.add_argument('--ddg', type=bool,  default=False,
                       help='rosetta ddg')
    parser.add_argument('--fake_msa', type=int,  default=None,
                       help='whether to use how many ProteinMPNN seqs to fake MSA')
    return parser.parse_args()

def convert_cif_to_pdb(cif_file, pdb_file):
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file)
        io = PDBIO()
        io.set_structure(structure)
        io.save(pdb_file)
        return True
    except Exception as e:
        print(f"convert error: {str(e)}")
        return False

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
        """init PyRosetta"""
        extra_options = "-mute all"
        if self.params_file:
            extra_options += f" -extra_res_fa {self.params_file}"
        pyrosetta.init(extra_options=extra_options)

    def _configure_analyzer(self):
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
        self.relax_protocol = FastRelax()
        self.relax_protocol.set_scorefxn(self.scorefxn)
        

        mm = MoveMap()
        mm.set_bb(True)
        mm.set_chi(True)
        self.relax_protocol.set_movemap(mm)
        # self.relax_protocol.set_script_file('MonomerRelax2019')

    def analyze_interface(self, pdb_path, protein_chain='A', ligand_chain='B'):
        try:
            self._init_pyrosetta()  

            pose = self._load_and_validate_pdb(pdb_path, protein_chain, ligand_chain)
            

            print("apply Relax")
            self.relax_protocol.apply(pose)
            

            self.analyzer.set_interface(f"{protein_chain}_{ligand_chain}")
            self.analyzer.apply(pose)
            return self._compile_results()
        except Exception as e:
            print(f"error: {e}")
            return None

    def _load_and_validate_pdb(self, pdb_path, protein_chain, ligand_chain):
        """load PD"""
        pose = pyrosetta.pose_from_pdb(pdb_path)

        chains = {pose.pdb_info().chain(res) for res in range(1, pose.total_residue()+1)}
        for chain in [protein_chain, ligand_chain]:
            if chain not in chains:
                raise ValueError(f"chain {chain} missing")

        lig_residues = [res for res in range(1, pose.total_residue()+1) 
                       if pose.pdb_info().chain(res) == ligand_chain]
        if not lig_residues:
            raise ValueError(f"chain {ligand_chain} is no ligand")
        print(f"load: {pdb_path}, total residues: {pose.total_residue()}")
        return pose

    def _compile_results(self):
        """collect data"""
        data = self.analyzer.get_all_data()
        return self.analyzer.get_interface_dG(), data.dSASA, data.packstat, data.interface_hbonds, data.sc_value,data.dG_dSASA_ratio, list(data.interface_residues)


def calculate_rosetta_metric(pdb_path):
    analyzer = InterfaceAnalyzer()


    interface_dG,dSASA,packstat,interface_hbonds,sc_value,energy_density,interface_residues = analyzer.analyze_interface(
        pdb_path=pdb_path,
        protein_chain='A',
        ligand_chain='B'
    )
    
    return interface_dG,dSASA,packstat,interface_hbonds

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
                protocol="binder", 
                initial_guess=True, 
                use_multimer=True,
                use_initial_atom_pos=False, num_recycles=af2_configs.num_recycles, 
                
                data_dir=af2_setting['params_dir'],
            )
    else:
        prediction_model = esm.pretrained.esmfold_v1()
        prediction_model = prediction_model.eval().cuda()
        prediction_model.set_chunk_size(256)
    return mpnn_model, mpnn_config_dict, prediction_model, af2_setting


def get_fake_msa(file_path: str,fake_msa,mpnn_model):
    mpnn_config_dict_for_msa = {
            "ca_only": True,
            "model_name": "v_48_020",
            'backbone_noise': 0.00,
            'temperature': 0.1,
            'num_seqs': fake_msa,
        }
    mpnn_config_dict_for_msa = OmegaConf.create(mpnn_config_dict_for_msa)
    mpnn_seqs, mpnn_scores = mpnn_design(
                config=mpnn_config_dict_for_msa,
                protein_path=file_path,
                model=mpnn_model,
                return_full_sequence=False,
                design_chains=['A']
            )
    fake_msa = ""
    msa = []
    for i, seq in enumerate(mpnn_seqs, 1):
        msa.append(f">Seq{i}")
        msa.append(seq)
    return "\n".join(msa)+"\n"


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

def calculate_average_b_factor(cif_file: str) -> float:

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("protein", cif_file)
        
        b_factors = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        b_factors.append(atom.bfactor)
        
        return sum(b_factors) / len(b_factors) if b_factors else 0.0
    except Exception as e:
        print(f"error: {str(e)}")
        return 0.0

def extract_chain_A(input_cif, output_cif):

    import copy
    parser = MMCIFParser()
    structure = parser.get_structure('input_structure', input_cif)

    model = structure[0]
    chain_a = copy.deepcopy(model['A'])
    new_structure = Structure.Structure('output')
    new_model = Model.Model(0)
    new_model.add(chain_a)
    new_structure.add(new_model)

    mmcif_io = MMCIFIO()
    mmcif_io.set_structure(new_structure)
    mmcif_io.save(output_cif)

def process_confidence_metrics(confidence_json_path: str, cif_path: str, copied_file : str) -> Dict:
    try:
        with open(confidence_json_path, 'r') as f:
            confidence_json = json.load(f)
        
        ptm = confidence_json['chain_ptm'][0]
        chain_pair_iptm = confidence_json['chain_pair_iptm'][0][1:]
        iptm = sum(chain_pair_iptm) / len(chain_pair_iptm) if chain_pair_iptm else 0
        
        chain_pair_pae_min = confidence_json['chain_pair_pae_min'][0][1:]
        ipae = sum(chain_pair_pae_min) / len(chain_pair_pae_min) if chain_pair_pae_min else 0
        
        plddt = calculate_average_b_factor(cif_path)
        rmsd = calculate_ca_rmsd(cif_path, copied_file)
        return {
            'AF3_PTM': ptm,
            'AF3_iPTM': iptm,
            'AF3_iPAE': ipae,
            'AF3_pLDDT': plddt,
            'AF3_rmsd': rmsd,
            'AF3_Status': 'Success'
        }
    except Exception as e:
        print(f"error: {str(e)}")
        return {
            'AF3_PTM': np.nan,
            'AF3_iPTM': np.nan,
            'AF3_iPAE': np.nan,
            'AF3_pLDDT': np.nan,
            'AF3_rmsd': np.nan,
            'AF3_Status': 'Failed'
        }
from Bio.PDB import MMCIFParser, PDBParser, Superimposer
from Bio.PDB.Atom import Atom
import numpy as np


from Bio.PDB import MMCIFParser, PDBParser, Superimposer

import numpy as np

def get_parser(filename):
    if filename.endswith(".cif"):
        return MMCIFParser(QUIET=True)
    elif filename.endswith(".pdb"):
        return PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

def get_first_model(structure):
    return next(structure.get_models())

def get_sorted_ca_atoms(model):
    atoms = []
    for chain in sorted(model, key=lambda c: c.id):  

        for residue in sorted(chain, key=lambda r: r.id[1]):

            if residue.has_id("CA"):
                atoms.append(residue["CA"])
    return atoms

def calculate_ca_rmsd(first_file, second_file,turn_over=False):
    try:

        first_parser = get_parser(first_file)
        second_parser = get_parser(second_file)
        

        struct1 = first_parser.get_structure("ref", first_file)
        struct2 = second_parser.get_structure("mobile", second_file)
        model1 = get_first_model(struct1)
        model2 = get_first_model(struct2)
        
        if turn_over:

            has_chain_a = 'A' in model1
            has_chain_b = 'B' in model1
            if has_chain_a and has_chain_b:
                chain_a = model1['A']
                chain_b = model1['B']
                

                model1.detach_child('A')
                model1.detach_child('B')
                
                chain_a.id = 'B'
                chain_b.id = 'A'

                model1.add(chain_b)
                model1.add(chain_a)


        ref_atoms = get_sorted_ca_atoms(model1)
        mobile_atoms = get_sorted_ca_atoms(model2)
        
        if len(ref_atoms) != len(mobile_atoms):
            raise ValueError("atoms incorrect")


        super_imposer = Superimposer()
        super_imposer.set_atoms(ref_atoms, mobile_atoms)  

        super_imposer.apply(mobile_atoms)

        
        return super_imposer.rms

    
    except Exception as e:
        print(f"rmsd failed: {str(e)}")
        return None

from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


from Bio.PDB.MMCIF2Dict import MMCIF2Dict

def template_process(cif_path, plddt_threshold=70, min_continuous_length=5):

    mmcif_dict = MMCIF2Dict(cif_path)

    b_factors = mmcif_dict["_atom_site.B_iso_or_equiv"]  
    chain_ids = mmcif_dict["_atom_site.auth_asym_id"]   

    residue_ids = mmcif_dict["_atom_site.auth_seq_id"]   


    b_factors = list(map(float, b_factors))


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

    chain_residue_avg_plddt = {}
    for chain_id, residue_dict in chain_residue_plddt.items():
        chain_residue_avg_plddt[chain_id] = {}
        for residue_id, plddt_list in residue_dict.items():
            avg_plddt = sum(plddt_list) / len(plddt_list)
            chain_residue_avg_plddt[chain_id][residue_id] = avg_plddt

    chain_to_residues = []
    for residue_id, avg_plddt in chain_residue_avg_plddt.get('A', {}).items():
        if avg_plddt > plddt_threshold:
            chain_to_residues.append(residue_id)
    

    filtered_indices = []
    chain_to_residues.sort()
    current_sequence = []
    for i, residue_id in enumerate(chain_to_residues):
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

    result = [{
        "mmcif": "",
        "queryIndices": filtered_indices,
        "templateIndices": filtered_indices

    }]
    print(filtered_indices)
    return result



def self_consistency(scaffold_path,output_path_tag, mpnn_model, mpnn_config_dict, prediction_model, af2_setting,plddt_good_indicates,multimer,ddg):
    
    mpnn_seqs, mpnn_scores = mpnn_design(
                config=mpnn_config_dict,
                protein_path=scaffold_path,
                model=mpnn_model,
                return_full_sequence=True,
                design_chains=['A'],
                fixed_residues=plddt_good_indicates,
            )
    #print(mpnn_seqs)
    scaffold_prot = from_pdb_string(open(scaffold_path).read(), 'A' if not multimer else None)
    length = scaffold_prot.atom_positions.shape[0]
    struct2seq_results = [{'mpnn_sequence': seq[:length], 'sequence_score': score} for seq, score in zip(mpnn_seqs, mpnn_scores)]
    struct2seq_df = pd.DataFrame(struct2seq_results)
    mpnn_seqs = struct2seq_df['mpnn_sequence'].values
    
    print(mpnn_seqs)
    results_list = []
    for i in range(len(mpnn_seqs)):
        chain_separator = np.where(scaffold_prot.chain_index[:-1] != scaffold_prot.chain_index[1:])[0] + 1
        chain_nums = chain_separator.shape[0] + 1
        first_chain_length = chain_separator[0] if chain_nums > 1 else length

        sequence = mpnn_seqs[i] + "".join([rc.restypes[i] for i in scaffold_prot.aatype.tolist()])[length:]
        sequence = list(sequence)
        for pos in reversed(chain_separator):
            sequence.insert(pos, ':')
        sequence = ''.join(sequence)
        print(sequence)
        evaluated_results, pred_prots = run_folding_and_evaluation(prediction_model, sequence, scaffold_prot, None, af2_setting, template_chains=["B"])
        #print(evaluated_results)
        for j, (result, pred_prot) in enumerate(zip(evaluated_results, pred_prots)):
            fold_path = os.path.join(os.path.dirname(scaffold_path), output_path_tag + f"_af2_{i}.pdb")
            with open(fold_path, 'w') as f:
                f.write(protein.to_pdb(pred_prot))
            result["rmsd_manual"] = calculate_ca_rmsd(fold_path,scaffold_path,turn_over=True)
            result["alphafold_path"] = fold_path
            result['mpnn_sequence'] = mpnn_seqs[i]
            result['sequence'] = sequence  
            result['index'] = i           
            print(ddg)
            if ddg:
                result["interface_dG"] ,result["dSASA"] ,result["packstat"],result["interface_hbonds"] = calculate_rosetta_metric(fold_path)
            else:
                result["interface_dG"] ,result["dSASA"] ,result["packstat"],result["interface_hbonds"] = 0,0,0,0
            results_list.append(result)
    return results_list

import subprocess

def run_alphafold3(json_path: str, pkl_path: str, output_dir: str,ref_time_steps: str,num_samples: str,) -> bool:
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
    #    print(f"error: {str(e)}")
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
                      reconstruct,
                      fake_msa,
                      multimer,
                      ddg,
                      run_af3: bool = True) -> Dict:
    metrics = {
        'PDB': pdb_file,
        'Cycle': cycle + 1,
        'AF2_RMSD': np.nan,
        'AF2_pLDDT': np.nan,
        'pae' : np.nan,
        'iptm' : np.nan, 
        'interface_dG' : np.nan,
        'dSASA'  : np.nan,
        'packstat' : np.nan,
        'interface_hbonds' : np.nan,
        'AF3_PTM': np.nan,
        'AF3_iPTM': np.nan,
        'AF3_iPAE': np.nan,
        'AF3_pLDDT': np.nan,
        'AF3_rmsd': np.nan,
        'AF2_Results' : [],
        'AF3_Status': 'Not Run'
    }
    
    try:
        target_dir = os.path.join(output_dir, f"recycle_{cycle+1}")
        os.makedirs(target_dir, exist_ok=True)
        

        source_file = os.path.join(input_dir, pdb_file) if cycle == 0 else input_dir
        copied_file = os.path.join(target_dir, pdb_file.replace(".pdb", f"_recycle_{cycle+1}.pdb"))

        shutil.copy(source_file, copied_file)
        plddt_good_indicates =[]
        if template_plddt_threshold > 0 and cycle >=1:
            tag_pre = f"{pdb_file}_recycle_{cycle}".lower()
            previous_dir = os.path.join(output_dir, f"recycle_{cycle}")
            cif_pre_path= os.path.join(previous_dir, tag_pre.replace(".pdb", ""), 
                                  tag_pre.replace(".pdb", "") + "_model.cif")
            template_to_json= template_process(cif_pre_path,template_plddt_threshold)
            plddt_good_indicates = template_to_json[0]["templateIndices"]
            print(plddt_good_indicates)

        results_af2 = self_consistency(
            copied_file, 
            pdb_file.replace(".pdb", f"_recycle_{cycle+1}"),
            mpnn_model, 
            mpnn_config_dict, 
            prediction_model, 
            af2_setting,
             [("A", resi) for resi in plddt_good_indicates],
             multimer,
             ddg
        )
        
        min_rmsd = float('inf')
        corresponding_plddt = None
        pae =None
        iptm =None 
        interface_dG = None 
        dSASA  = None 
        packstat = None 
        interface_hbonds =None 
        for entry in results_af2:
            if entry['rmsd_manual'] < min_rmsd:
                min_rmsd = entry['rmsd_manual']
                corresponding_plddt = entry['plddt']
                iptm = entry["i_ptm"]
                pae = entry["pae"]
                interface_dG = entry["interface_dG"] 
                dSASA  = entry["dSASA"] 
                packstat = entry["packstat"]
                interface_hbonds  = entry["interface_hbonds"] 
        metrics['AF2_RMSD'] = min_rmsd
        metrics['AF2_pLDDT'] = corresponding_plddt
        metrics["pae"] = pae
        metrics["iptm"] = iptm 
        metrics["interface_dG"] =interface_dG
        metrics["dSASA"] = dSASA
        metrics["packstat"] = packstat
        metrics["interface_hbonds"] =interface_hbonds
        for entry in results_af2:
            metrics['AF2_Results'].append({
                'sequence': entry['sequence'],
                'rmsd': entry['rmsd_manual'],
                'plddt': entry['plddt'],
                "iptm" : entry["i_ptm"],
                "pae" : entry["pae"],
                "interface_dG ": entry["interface_dG"] ,
                "dSASA" : entry["dSASA"] ,
                "packstat" : entry["packstat"] ,
                "interface_hbonds" :  entry["interface_hbonds"] 
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
            sequence_string = [entry['sequence'] for entry in results_af2 if 'alphafold_path' in entry][0]
            output_file = copied_file.replace(".pdb","_pose_seq.pdb")
            pose_sequence.process(copied_file, sequence_string, output_file)
            copied_file = output_file
        else:
            copied_file = copied_file
            
        input_json = template.copy()
        input_json['name'] = tag.replace(".pdb", "")
        input_json['sequences'][0]['protein']["sequence"] = get_chain_sequence(copied_file, 'A')
        #input_json['sequences'][1]['protein']["sequence"] = get_chain_sequence(copied_file, 'B')
        input_json["modelSeeds"] = get_random_seeds(num_seeds)
        if fake_msa and cycle >=1:
            tag_pre = f"{pdb_file}_recycle_{cycle}".lower()
            previous_dir = os.path.join(output_dir, f"recycle_{cycle}")
            seeds_path = os.path.join(previous_dir, tag_pre.replace(".pdb", ""))
            subfolders = [f for f in os.listdir(seeds_path) 
                 if os.path.isdir(os.path.join(seeds_path, f))]
            msa=f'>query\n{input_json["sequences"][0]["protein"]["sequence"]}\n'
            for subfolder in subfolders:
                pdb_file = os.path.join(seeds_path, subfolder,subfolder+"_model.cif")
                msa_chunk = get_fake_msa(pdb_file, fake_msa, mpnn_model)
                msa += msa_chunk
            input_json['sequences'][0]['protein']["unpairedMsa"] = msa
        if template_plddt_threshold > 0 and cycle >=1:
            A_chain_path = cif_pre_path.replace(".cif","chain_A.cif")
            extract_chain_A(cif_pre_path, A_chain_path)
            template_to_json[0]["mmcif"] = _read_file(pathlib.Path(A_chain_path), pathlib.Path(json_path))
            input_json['sequences'][0]['protein']['templates'] = template_to_json
        with open(json_path, 'w') as f:
            json.dump(input_json, f, indent=2)
        
        # pkl_process
        result, file_name, error = process_single_file(( copied_file, target_dir, target_dir))
        
        pkl_path = os.path.join(target_dir, f'{os.path.splitext(copied_file)[0]}.pkl')
        clear_gpu_memory()
        success = run_alphafold3(json_path, pkl_path, target_dir, ref_time_steps=str(ref_time_steps), num_samples=str(num_samples))
        
        if success:
            tag = f"{tag}".lower()
            cif_path = os.path.join(target_dir, tag.replace(".pdb", ""), 
                                  tag.replace(".pdb", "") + "_model.cif")
            confidence_path = os.path.join(target_dir, tag.replace(".pdb", ""), 
                                         tag.replace(".pdb", "") + "_summary_confidences.json")
            print(cif_path,tag)
            af3_metrics = process_confidence_metrics(confidence_path, cif_path, copied_file)
            metrics.update(af3_metrics)

            pdb_output = cif_path.replace(".cif", ".pdb")
            if convert_cif_to_pdb(cif_path, pdb_output):
                return metrics, pdb_output
            else:
                print(f"convert failed")
                return metrics, copied_file
        else:
            print(f"AF3 error")
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
                    if metrics['AF3_pLDDT'] > args.early_stop_threshold and metrics['AF3_rmsd'] < 1:
                        is_last_cycle =True
                
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
                    args.multimer,
                    args.ddg, 
                    run_af3=not is_last_cycle 
                )
                all_results.append(metrics)
                current_input = next_input 
                
                print(f"  Cycle {cycle+1} completed:")
                print(f"    AF2 RMSD: {metrics['AF2_RMSD']:.3f}")
                print(f"    AF2 pLDDT: {metrics['AF2_pLDDT']:.3f}")
                print(f"    AF3 Status: {metrics['AF3_Status']}")
                if metrics['AF3_Status'] == 'Success':
                    print(f"    AF3 PTM: {metrics['AF3_PTM']:.3f}")
                    print(f"    AF3 pLDDT: {metrics['AF3_pLDDT']:.3f}")
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
