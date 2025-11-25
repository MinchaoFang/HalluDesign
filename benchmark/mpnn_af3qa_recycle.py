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
from Bio.PDB import PDBParser, MMCIFParser
import json
import csv
import os
import sys
sys.path.insert(0,"/storage/caolongxingLab/fangminchao/Proteus/Proteus_flow_matching")
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
        print(f"error: {str(e)}")
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
                design_chains=['A']
            )
    fake_msa = ""
    msa = []
    for i, seq in enumerate(mpnn_seqs, 1):
        msa.append(f">Seq{i}")
        msa.append(seq)
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

def calculate_ca_rmsd(cif_file, pdb_file):

    cif_parser = MMCIFParser(QUIET=True)
    cif_structure = cif_parser.get_structure("CIF", cif_file)

    pdb_parser = PDBParser(QUIET=True)
    pdb_structure = pdb_parser.get_structure("PDB", pdb_file)

    cif_ca_atoms = get_ca_atoms(cif_structure)
    pdb_ca_atoms = get_ca_atoms(pdb_structure)


    if len(cif_ca_atoms) != len(pdb_ca_atoms):
        raise ValueError("CIf and PDB atoms incorrect")


    cif_coords = np.array([atom.coord for atom in cif_ca_atoms])
    pdb_coords = np.array([atom.coord for atom in pdb_ca_atoms])

    super_imposer = Superimposer()
    super_imposer.set_atoms(pdb_ca_atoms, cif_ca_atoms)
    super_imposer.apply(pdb_ca_atoms)  

    # return RMSD
    return super_imposer.rms

from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

def template_process(cif_path, plddt_threshold=70, min_continuous_length=5):
    """
    Extracts continuous high-confidence regions (based on pLDDT values)
    from an AlphaFold or predicted structure in mmCIF format.
    
    Args:
        cif_path (str): Path to the mmCIF file.
        plddt_threshold (float): Minimum per-residue pLDDT score to be considered reliable.
        min_continuous_length (int): Minimum number of consecutive residues above the threshold to keep.

    Returns:
        list[dict]: A list containing one dictionary with the following keys:
            - "mmcif": Placeholder (empty string)
            - "queryIndices": Residue indices that passed the filtering criteria
            - "templateIndices": Same as queryIndices
    """
    
    # Read the mmCIF file
    mmcif_dict = MMCIF2Dict(cif_path)
    
    # Extract per-atom pLDDT (stored as B-factor), chain IDs, and residue IDs
    b_factors = mmcif_dict["_atom_site.B_iso_or_equiv"]   # pLDDT values
    chain_ids = mmcif_dict["_atom_site.auth_asym_id"]     # Chain IDs
    residue_ids = mmcif_dict["_atom_site.auth_seq_id"]    # Residue IDs
    
    # Convert pLDDT values to float
    b_factors = list(map(float, b_factors))
    
    # Store per-residue pLDDT values grouped by chain
    chain_residue_plddt = {}
    for i in range(len(b_factors)):
        chain_id = chain_ids[i]
        residue_id = int(residue_ids[i]) - 1  # Convert to 0-based index
        plddt = b_factors[i]
        
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
    
    # Identify residues above the pLDDT threshold
    chain_to_residues = {}
    for chain_id, residue_dict in chain_residue_avg_plddt.items():
        for residue_id, avg_plddt in residue_dict.items():
            if avg_plddt > plddt_threshold:
                if chain_id not in chain_to_residues:
                    chain_to_residues[chain_id] = []
                chain_to_residues[chain_id].append(residue_id)
    
    # Filter for continuous residue segments
    filtered_indices = []

    all_indices = []
    for chain_id, residues in chain_to_residues.items():
        all_indices.extend(residues)  # Collect all residues passing the threshold
    
    # Identify continuous regions that meet minimum length requirement
    for chain_id, residues in chain_to_residues.items():
        residues.sort()
        current_sequence = []
        for i, residue_id in enumerate(residues):
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

    # Build the final result dictionary
    result = [{
        "mmcif": "",
        "queryIndices": filtered_indices,
        "templateIndices": filtered_indices
    }]
    
    return result



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
    """run AlphaFold3"""
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
    #    print(f"out of time: {str(e)}")
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
        return False

    except subprocess.TimeoutExpired as e:
        print("Command timed out!")
        print("Timeout:", e.timeout)
        return False

    except Exception as e:
        print("An unexpected error occurred:", str(e))
        return False

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
                      run_af3: bool = True) -> Dict:

    metrics = {
        'PDB': pdb_file,
        'Cycle': cycle + 1,
        'AF2_RMSD': np.nan,
        'AF2_pLDDT': np.nan,
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
            tag_pre = f"{pdb_file}_recycle_{cycle}"
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
             [("A", resi) for resi in plddt_good_indicates]
        )
        
        min_rmsd = float('inf')
        corresponding_plddt = None
        seq = None
        for entry in results_af2:
            if entry['rmsd'] < min_rmsd:
                min_rmsd = entry['rmsd']
                corresponding_plddt = entry['plddt']
                seq = entry['sequence']
        
        metrics['AF2_RMSD'] = min_rmsd
        metrics['AF2_pLDDT'] = corresponding_plddt
        
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
            sequence_string = [entry['sequence'] for entry in results_af2 if 'alphafold_path' in entry][0]
            output_file = copied_file.replace(".pdb","_pose_seq.pdb")
            pose_sequence.process(copied_file, sequence_string, output_file)
            copied_file = output_file
        else:
            copied_file = copied_file
            
        input_json = template.copy()
        input_json['name'] = tag.replace(".pdb", "")
        input_json['sequences'][0]['protein']["sequence"] = get_chain_sequence(copied_file, 'A')
        input_json["modelSeeds"] = get_random_seeds(num_seeds)
        if fake_msa and cycle >=1:
            tag_pre = f"{pdb_file}_recycle_{cycle}"
            previous_dir = os.path.join(output_dir, f"recycle_{cycle}")
            seeds_path = os.path.join(previous_dir, tag_pre.replace(".pdb", ""))
            subfolders = [f for f in os.listdir(seeds_path) 
                 if os.path.isdir(os.path.join(seeds_path, f))]
            msa=""
            for subfolder in subfolders:
                pdb_file = os.path.join(seeds_path, subfolder,subfolder+"_model.cif")
                msa=msa+get_fake_msa(pdb_file,fake_msa,mpnn_model)
            input_json['sequences'][0]['protein']["pairedMsa"] = msa
        if template_plddt_threshold > 0 and cycle >=1:
            template_to_json[0]["mmcif"] = _read_file(pathlib.Path(cif_pre_path), pathlib.Path(json_path))
            input_json['sequences'][0]['protein']['templates'] = template_to_json
        with open(json_path, 'w') as f:
            json.dump(input_json, f, indent=2)
        
        # pkl_process
        result, file_name, error = process_single_file(( copied_file, target_dir, target_dir))
        
        pkl_path = os.path.join(target_dir, f'{os.path.splitext(copied_file)[0]}.pkl')
        clear_gpu_memory()
        success = run_alphafold3(json_path, pkl_path, target_dir, ref_time_steps=str(ref_time_steps), num_samples=str(num_samples))
        
        if success:
            cif_path = os.path.join(target_dir, tag.replace(".pdb", ""), 
                                  tag.replace(".pdb", "") + "_model.cif")
            confidence_path = os.path.join(target_dir, tag.replace(".pdb", ""), 
                                         tag.replace(".pdb", "") + "_summary_confidences.json")
            

            af3_metrics = process_confidence_metrics(confidence_path, cif_path, copied_file)
            metrics.update(af3_metrics)

            pdb_output = cif_path.replace(".cif", ".pdb")
            if convert_cif_to_pdb(cif_path, pdb_output):
                return metrics, pdb_output
            else:
                print(f"convert faield")
                return metrics, copied_file
        else:
            print(f"AF error")
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
