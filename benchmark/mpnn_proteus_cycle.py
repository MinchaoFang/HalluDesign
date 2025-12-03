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
    print(mpnn_seqs)
    fake_msa = ""
    msa = []
    for i, seq in enumerate(mpnn_seqs, 1):
        msa.append(f">Seq{i}")
        msa.append(seq)
    return "\n".join(msa)


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
    pdb_parser = PDBParser(QUIET=True)
    cif_structure = pdb_parser.get_structure("PDB", cif_file)

    pdb_parser = PDBParser(QUIET=True)
    pdb_structure = pdb_parser.get_structure("PDB", pdb_file)

    cif_ca_atoms = get_ca_atoms(cif_structure)
    pdb_ca_atoms = get_ca_atoms(pdb_structure)

    if len(cif_ca_atoms) != len(pdb_ca_atoms):
        raise ValueError("atoms incorrect, unable to get RMSD")

    cif_coords = np.array([atom.coord for atom in cif_ca_atoms])
    pdb_coords = np.array([atom.coord for atom in pdb_ca_atoms])

    super_imposer = Superimposer()
    super_imposer.set_atoms(pdb_ca_atoms, cif_ca_atoms)
    super_imposer.apply(pdb_ca_atoms)  # aligan

    # return RMSD
    return super_imposer.rms



def self_consistency(scaffold_path,output_path_tag, mpnn_model, mpnn_config_dict, prediction_model, af2_setting):
    
    mpnn_seqs, mpnn_scores = mpnn_design(
                config=mpnn_config_dict,
                protein_path=scaffold_path,
                model=mpnn_model,
                design_chains=['A']
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

def run_proteus_partial(file_path: str, output_dir: str,ref_time_steps: str,) -> bool:
    """ RFD partial diffusion"""

    parser = PDBParser()
    structure = parser.get_structure('PDB_structure', file_path)
    
    # lens for RFD
    residues = [residue for residue in structure.get_residues() if residue.get_id()[0] == ' ']
    length = len(residues)

    try:
    
        # get name
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        print()
        command = [
            "python", "/storage/caolongxingLab/fangminchao/tools/Proteus_flow_matching/experiments/inference_se3_flows.py",
            "inference.self_consistency.enable=False", "inference.ckpt_path='/storage/caolongxingLab/wangchentong/Project/Proteus_flow_matching/weights/codesign_dplm_long/epoch\=41-step\=1500000.ckpt'",
            "interpolant.sampling.num_timesteps=100", "inference.samples.sample_start_idx=0",
            "inference.model_trace_dir=False", "inference.samples.sample_num=1",
            "inference.name=''",
            "interpolant.t_schedule=linear",
            f"interpolant.sampling.partial_timesteps={ref_time_steps}",
            f"inference.samples.ref_pdb='{file_path}'", 
            f"inference.samples.contigs='A1-{length}'", 
            f"inference.samples.partial_contigs='A1-{length}'", 
            f"inference.output_dir={os.path.join(output_dir, file_name)}",
            f"inference.samples.prefix='{file_name}'"
        ]
        cmd_str = " ".join(command)
        print("Executing command:", cmd_str) 
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
    #    print(f"RFD error: {str(e)}")
    #    return False
    #    # Run the command
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


def process_single_pdb(pdb_file: str, 
                      cycle: int,
                      input_dir: str,
                      output_dir: str,
                      mpnn_model,
                      mpnn_config_dict,
                      prediction_model,
                      af2_setting,
                      ref_time_steps,
                      reconstruct,
                      fake_msa,
                      run_af3: bool = True) -> Dict:

    metrics = {
        'PDB': pdb_file,
        'Cycle': cycle + 1,
        'AF2_RMSD': np.nan,
        'AF2_pLDDT': np.nan,
        'AF2_Results' : [],
        'RFD_RMSD' : np.nan,
        'RFD_Status' : 'Not run'
    }
    
    try:
        target_dir = os.path.join(output_dir, f"recycle_{cycle+1}")
        os.makedirs(target_dir, exist_ok=True)
        

        source_file = os.path.join(input_dir, pdb_file) if cycle == 0 else input_dir
        copied_file = os.path.join(target_dir, pdb_file.replace(".pdb", f"_recycle_{cycle+1}.pdb"))

        shutil.copy(source_file, copied_file)

        results_af2 = self_consistency(
            copied_file, 
            pdb_file.replace(".pdb", f"_recycle_{cycle+1}"),
            mpnn_model, 
            mpnn_config_dict, 
            prediction_model, 
            af2_setting
        )
        
        min_rmsd = float('inf')
        corresponding_plddt = None
        for entry in results_af2:
            if entry['rmsd'] < min_rmsd:
                min_rmsd = entry['rmsd']
                corresponding_plddt = entry['plddt']
        
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
            
        clear_gpu_memory()
        success = run_proteus_partial(copied_file, target_dir, ref_time_steps=str(ref_time_steps))
        
        if success:
            metrics['RFD_Status'] = 'success'
            file_name = os.path.splitext(os.path.basename(copied_file))[0]
            # convert cif to pdb
            pdb_output = os.path.join(target_dir, file_name ,"scaffold", file_name)+"_sample_0_scaffold.pdb"
            print(copied_file, pdb_output)
            metrics['RFD_RMSD'] = calculate_ca_rmsd(copied_file, pdb_output)
            print(metrics['RFD_RMSD'])
            return metrics, pdb_output
        else:
            print("RFD processing failed, continuing with the original file.")
            metrics['RFD_Status'] = 'Failed'
            return metrics, copied_file
            
    except Exception as e:
        print(f"error: {str(e)}")
        return metrics, copied_file


import pandas as pd
import os
from filelock import FileLock

def main():
    args = parse_arguments()
    
    
    # init
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
                metrics, next_input = process_single_pdb(
                    pdb_file,
                    cycle,
                    current_input,
                    args.output_dir,
                    mpnn_model,
                    mpnn_config_dict,
                    prediction_model,
                    af2_setting,
                    args.ref_time_steps,
                    args.reconstruct,
                    args.fake_msa,
                    run_af3=not is_last_cycle  
                )
                all_results.append(metrics)
                current_input = next_input 
                
                print(f"  Cycle {cycle+1} completed:")
                print(f"    AF2 RMSD: {metrics['AF2_RMSD']:.3f}")
                print(f"    AF2 pLDDT: {metrics['AF2_pLDDT']:.3f}")
                print(f"    RFD Status: {metrics['RFD_Status']}")

                with lock:
                    file_exists = os.path.exists(csv_path)
                    pd.DataFrame([metrics]).to_csv(csv_path, mode='a', header=not file_exists, index=False)
                    
            except Exception as e:
                print(f"  Error in cycle {cycle+1}: {str(e)}")
                continue
    
    print(f"Processing completed. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
