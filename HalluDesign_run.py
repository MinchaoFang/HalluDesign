import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Disable pre-allocating the entire GPU memory
# os.environ['XLA_CLIENT_MEM_FRACTION'] = '0.5'          # Limit GPU memory usage to 50%
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import ast
import re
import argparse
import pandas as pd
import sys
print(sys.path)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from pathlib import Path
import json
import shutil
from Bio.PDB import MMCIFParser, PDBParser, PDBIO
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
import os
import copy
import shutil
from data.utility import *
from eval.evaluation import  CoDP

from eval.eval_utility import get_random_seeds,generate_metrics,get_global_residue_index
from LigandMPNN.package import MPNNModel
from af3_model import AF3DesignerPack
import subprocess
import math
from copy import deepcopy
import csv
import torch
from rdkit import Chem
from rdkit.Chem import rdMolAlign
import numpy as np
import pandas as pd
import os
from filelock import FileLock
import string
from models_utility import *

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process multiple PDB files with AF3 based optimizer')
    parser.add_argument('--pdb_list', type=str, required=False,
                       help='Path to text file containing list of PDB files')
    parser.add_argument('--input_file', type=str, required=False,
                       help='input file path')
    parser.add_argument('--fix_res_index', type=str, required=False, 
                        help='Fixed residue indices, e.g. A1 B4 but be careful, we will reindex all to begin with 1')
    parser.add_argument('--fix_chain_index', type=str, required=False,
                        help='Fixed chain indices, e.g. A B')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory for output files')
    parser.add_argument('--num_seqs', type=int, default=8,
                       help='Number of proteinMPNN or LigadnMPNN+proteinMPNN seqs to perform self consistency')
    parser.add_argument('--num_recycles', type=int, default=10,
                       help='toatl Number of recycles to perform')
    parser.add_argument('--ref_time_steps', type=int, default=50,
                       help='ref time steps to perform')
    parser.add_argument('--cdr',  type=str,  required=False,
                    help='cdr input to refine antibody,should use in antibody')
    parser.add_argument('--fix_seq_file',  type=str,  required=False,
                    help='fix seq csv which contains file_path and fix res index')
    parser.add_argument('--framework_seq',  type=str, nargs='+', required=False, default=[],
                    help='framework seq input to refine antibody,sequence defined in this will be fixed')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='AF3 optimizer batchsize')
    parser.add_argument('--num_seeds', type=int, default=1,
                       help='AF3 number random seeds')
    parser.add_argument('--early_stop_threshold', type=int, default=0,
                       help='AF3 early_stop_plddt_threshold')
    parser.add_argument('--template_plddt_threshold', type=int, default=0,
                       help='AF3 template_plddt_threshold')
    parser.add_argument('--template_plddt_threshold_length', type=int, default=5,
                       help='AF3 template_plddt_threshold length _threshold')
    parser.add_argument('--template_path', type=str, required=True,
                       help='Path to AF3 template.json file, it should be really careful to treat with')
    parser.add_argument('--template_for_eval', type=str, required=False, 
                       help='Path to AF3 template.json file only for eval')
    parser.add_argument('--prediction_model', type=str,  required=True,
                       help='af3 or Protenix')
    parser.add_argument('--ref_eval', action='store_true', default=False,
                    help='whether to use small molecular ref position in AF3 evaluation, which means direct send atom positions to AF3, only use it confidence head')
    parser.add_argument('--esmhead', action='store_true', default=False,
                    help='whether to use esmhead to validate sequence quality')
    parser.add_argument('--fake_msa', type=int,  default=None,
                       help='whether to use how many ProteinMPNN (for pure protien system) or LigandMPNN (for protien and ligand system) seqs to fake MSA')
    parser.add_argument('--sm', type=str, nargs='+', required=False, default=[],
                    help='smiles input,it needs a good structure mapping smilles')
    parser.add_argument('--mpnn',  type=str,  required=False,
                    help='which mpnn model do you choose proteinmpnn ligandmpnn ligandmpnn_plus_proteinmpnn')
    parser.add_argument('--mpnn_temperature', type=float, default=0.1,
                       help='mpnn temperature to use')
    parser.add_argument('--replace_MSA', action='store_true', default=False,
                       help='framework to use MSA, used in antibody design')
    parser.add_argument('--ccd', type=str, nargs='+', required=False, default=[],
                    help='ccd input,it needs a good structure mapping ccd')
    parser.add_argument('--dna', type=str, nargs='+', required=False, default=[],
                    help='dna input')
    parser.add_argument('--rna', type=str, nargs='+', required=False, default=[],
                    help='rna input')
    parser.add_argument('--design_epoch_begin', type=int, required=False,  default=0,
                    help='in which cycles, multi batch evaluation process will begin')
    parser.add_argument("--symmetry_residues",type=str,default="",
                    help="Add list of res for which residues need to be symmetric, e.g. 'A12,A13,A14|C2,C3|A5,B6'")
    parser.add_argument("--symmetry_chains",type=str,default="",
                    help="Add list of chains for which chains need to be symmetric, e.g. 'A,B'")
    parser.add_argument("--pocket_fix", action='store_true', default=False,
                    help="for repeat sample")
    parser.add_argument("--cyclic", type=int, default=None,
                    help="for cyclic peptide design, only for peptide A")
    parser.add_argument("--ptm",  type=str, nargs='+', required=False, default=[],
                    help="for ptm design")
    parser.add_argument("--random_init", action='store_true', default=False,
                    help="for pure noise generation")
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    if not os.path.exists(args.template_path):
        raise FileNotFoundError(f"Template file {args.template_path} not found!")
    
    if args.pdb_list and args.input_file:
        raise ValueError("Cannot specify both --pdb_list and --input_file at the same time")

    if args.sm and args.ccd:
        raise ValueError("Cannot specify both --sm and --ccd at the same time, only allow one")
    
    if args.symmetry_residues and args.symmetry_chains:
        raise ValueError("Cannot specify both --symmetry_residues and --symmetry_chains at the same time, only allow one")

    if args.pdb_list:
        with open(args.pdb_list, 'r') as f:
            pdb_files = [line.strip() for line in f if line.strip()]
    elif args.input_file:
        pdb_files = [args.input_file]
    else:
        raise ValueError("You must specify either --pdb_list or --input_file")
        
    if args.sm or args.dna or args.rna or args.ccd:
        if args.mpnn == "protein_mpnn":
            mpnn_model = MPNNModel(model_name="protein_mpnn",
                             T=args.mpnn_temperature, 
                             ligand_mpnn_use_side_chain_context=1,
                             ligand_mpnn_use_atom_context=1,
                             number_of_packs_per_design=1,
                             pack_side_chains=1,
                             parse_atoms_with_zero_occupancy=1,
                             pack_with_ligand_context= 0,
                             repack_everything=1)
            #raise ValueError("protein_mpnn can not use in small molecular or dna or rna system design")
        if args.mpnn == "ligand_mpnn":
            mpnn_model = MPNNModel(model_name=args.mpnn,
                             T=args.mpnn_temperature, 
                             ligand_mpnn_use_side_chain_context=1,
                             ligand_mpnn_use_atom_context=1,
                             number_of_packs_per_design=1,
                             pack_side_chains=1,
                             parse_atoms_with_zero_occupancy=1,
                             pack_with_ligand_context= 0,
                             repack_everything=1)
        elif args.mpnn == "ligandmpnn_plus_proteinmpnn":
            Ligand_mpnn_model = MPNNModel(model_name="ligand_mpnn",
                             T=args.mpnn_temperature, 
                             ligand_mpnn_use_side_chain_context=1,
                             ligand_mpnn_use_atom_context=1,
                             number_of_packs_per_design=1,
                             pack_side_chains=1,
                             parse_atoms_with_zero_occupancy=1,
                             pack_with_ligand_context= 0,
                             repack_everything=1)
            Protein_mpnn_model = MPNNModel(model_name="soluble_mpnn",
                             T=args.mpnn_temperature, 
                             ligand_mpnn_use_side_chain_context=1,
                             ligand_mpnn_use_atom_context=1,
                             number_of_packs_per_design=1,
                             pack_side_chains=1,
                             parse_atoms_with_zero_occupancy=1,
                             pack_with_ligand_context= 0,
                             repack_everything=1)
            mpnn_model= [Ligand_mpnn_model,Protein_mpnn_model]
        
    else:
        mpnn_model = MPNNModel(model_name="protein_mpnn",
                         T=args.mpnn_temperature, 
                         ligand_mpnn_use_side_chain_context=1,
                         ligand_mpnn_use_atom_context=1,
                         number_of_packs_per_design=1,
                         pack_side_chains=1,
                         parse_atoms_with_zero_occupancy=1,
                         pack_with_ligand_context= 0,
                         repack_everything=1)
    mpnn_config_dict = {
        "temperature": args.mpnn_temperature,
        "model_name": args.mpnn, #"ligandmpnn_plus_proteinmpnn"
        "num_seqs": 1
        }
    print(f"{args.mpnn} will be use in sequnce design")
    if args.fix_res_index:
        fixed_residues = args.fix_res_index.split()
    else:
        fixed_residues = []
    evaluator = None
    if args.esmhead:
        checkpoints_to_run = "./CoDP/ckpt/epoch_1_without_esm2.pth"
        esm_name = "facebook/esm2_t33_650M_UR50D"
        #! remain to do!
        evaluator = CoDP(checkpoints_to_run,esm_name)
    if args.fix_chain_index:
        fixed_chains = args.fix_chain_index.split()
    else:
        fixed_chains = []

    if args.prediction_model  == "af3":
        Designer_model = AF3DesignerPack(jax_compilation_dir=os.path.join(args.output_dir,"jax_compilation_cache_dir"))
        protein_chains, ligand_chains, dna_chains, rna_chains, chain_types =  count_chain_based_on_json(args.template_path)
        metrics = generate_metrics(protein_chains,ligand_chains, dna_chains, rna_chains,chain_types)
    if args.prediction_model  == "protenix":
        sys.path.insert(0,os.path.join(current_dir,"Protenix"))
        from runner.inference import ProtenixInferrer
        os.environ["LAYERNORM_TYPE"] = "fast_layernorm"
        os.environ["USE_DEEPSPEED_EVO_ATTENTION"] = "true"
        os.environ["CUTLASS_PATH"] = "./cutlass" 
        static_configs = {
        "model.N_cycle": 10,
        "sample_diffusion.N_sample": 5,
        "sample_diffusion.N_step": 200, # Example value
        "use_esm": True,
        "use_msa": False,
        # Add any other global or model-specific configs here
        # "load_checkpoint_path": "/path/to/your/model_checkpoint.pth", # Ensure this is correct
        # "need_atom_confidence": True, # Or False, depending on your needs
        # "sorted_by_ranking_score": True, # Or False
        # "dtype": "fp32", # or "bf16", "fp16"
            }

        print("Initializing ProtenixInferrer...")
        Designer_model = ProtenixInferrer(cyclic = args.cyclic,**static_configs)
        print("ProtenixInferrer initialized.")
        protein_chains, ligand_chains, dna_chains, rna_chains, chain_types =  count_chain_based_on_json_protenix(args.template_path)
        metrics = generate_metrics(protein_chains,ligand_chains, dna_chains, rna_chains,chain_types)
    # use lock file to process 
    csv_path = os.path.join(args.output_dir, 'processing_results.csv')
    lock_path = f"{csv_path}.lock" 
    lock = FileLock(lock_path)
    metrics_new =  copy.deepcopy(metrics)
    all_results = []
    for pdb_file in pdb_files:
        print(f"\nProcessing {pdb_file}...")
        current_input = pdb_file
        chain_number_list_cdr = []
        bais_per_residues = None
        if  args.fix_seq_file:
            df_fix = pd.read_csv(args.fix_seq_file)
            if "fix_res" in df_fix.columns:
                df_fix_file = df_fix[df_fix['file_path']==os.path.basename(pdb_file)]["fix_res"]
            
                if not df_fix_file.empty:
                    fixed_residues = str(df_fix_file.values[0]).split()
            else:
                fixed_residues = []
            if "bias" in df_fix.columns:
                bias = df_fix[df_fix["file_path"].apply(os.path.basename) == os.path.basename(pdb_file)]["bias"]
    
                if not bias.empty:
                    bais_per_residues = ast.literal_eval(bias.values[0])
                    
            
        for cycle in range(args.num_recycles):
            print(f"  Starting cycle {cycle+1}")
            try:
                is_last_cycle = (cycle == args.num_recycles - 1)
                # early stop if you wish 
                if cycle > 0 and  args.early_stop_threshold > 0:
                    if metrics[0]['op_plddt'] > args.early_stop_threshold :
                        is_last_cycle =True
                design_begin = False
                if cycle >= args.design_epoch_begin:
                    mpnn_config_dict['num_seqs'] = args.num_seqs
                    design_begin = True
                print(f"begin multi-batch evaluation {design_begin}")
                metrics = copy.deepcopy(metrics_new)
                if args.prediction_model  == "af3":
                    metrics, next_input ,chain_number_list_cdr= af3_op_af3_eval(
                    current_input,
                    cycle,
                    args.output_dir,
                    args.template_path,
                    args.template_for_eval,
                    mpnn_model,
                    mpnn_config_dict,
                    Designer_model,
                    args.ref_time_steps,
                    args.num_samples,
                    args.num_seeds,
                    args.template_plddt_threshold,
                    args.template_plddt_threshold_length,
                    args.fake_msa,
                    args.ref_eval,
                     chain_types,
                     fixed_chains,
                     fixed_residues,
                     bais_per_residues,
                    metrics,
                    args.symmetry_residues,
                    args.symmetry_chains,
                    args.sm,
                    args.ccd,
                    args.dna,
                    args.rna,
                    args.cdr,
                    args.random_init,
                    args.framework_seq,
                    evaluator,
                    design_begin,
                    chain_number_list_cdr,
                    args.cyclic,
                    args.replace_MSA,
                    args.ptm,
                    run_af3=not is_last_cycle  #  AF3 not run in last cycle
                )
                elif args.prediction_model  == "protenix":
                    metrics, next_input ,chain_number_list_cdr= protenix_op_protenix_eval(
                    current_input,
                    cycle,
                    args.output_dir,
                    args.template_path,
                    args.template_for_eval,
                    mpnn_model,
                    mpnn_config_dict,
                    Designer_model,
                    args.ref_time_steps,
                    args.num_samples,
                    args.num_seeds,
                    args.ref_eval,
                     chain_types,
                     fixed_chains,
                     fixed_residues,
                     bais_per_residues,
                    metrics,
                    args.ptm,
                    args.symmetry_residues,
                    args.symmetry_chains,
                    args.sm,
                    args.ccd,
                    args.dna,
                    args.rna,
                    args.cdr,
                    args.random_init,
                    args.framework_seq,
                    evaluator,
                    design_begin,
                    chain_number_list_cdr,
                    args.cyclic,
                    args.replace_MSA,
                    run_af3=not is_last_cycle  #  AF3 not run in last cycle
                )
                all_results.append(metrics)
                current_input = next_input  # update for next cycle

                # save every cycle data
                with lock:
                    file_exists = os.path.exists(csv_path)
                    pd.DataFrame(metrics).to_csv(csv_path, mode='a', header=not file_exists, index=False)
                if is_last_cycle:
                   break 
            except Exception as e:
                print(f"  Error in cycle {cycle+1}: {str(e)}")
                continue
    print(f"Processing completed. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
