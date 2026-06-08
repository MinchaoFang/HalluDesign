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
import random
from data.utility import *
from eval.evaluation import  CoDP
from eval.eval_utility import generate_metrics
from LigandMPNN.package import MPNNModel
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

PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process multiple PDB files with AF3 based optimizer')
    parser.add_argument('--pdb_list', type=str, required=False,
                       help='Path to text file containing list of PDB files')
    parser.add_argument('--input_file', type=str, required=False,
                       help='input file path')
    parser.add_argument('--fix_res_index', type=str, required=False, 
                        help='Fixed residue indices, e.g. A1 B4 but be careful, you should reindex all chain to begin with 1')
    parser.add_argument('--fix_chain_index', type=str, required=False,
                        help='Fixed chain indices, e.g. A B')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory for output files')
    parser.add_argument('--num_seqs', type=int, default=8,
                       help='Number of proteinMPNN or LigadnMPNN seqs to perform self consistency')
    parser.add_argument('--num_recycles', type=int, default=10,
                       help='toatl Number of recycles to perform')
    parser.add_argument('--ref_time_steps', type=int, default=50,
                       help='ref time steps to perform')
    parser.add_argument('--cdr',  type=str,  required=False,
                    help='cdr input to refine antibody,should use in antibody')
    parser.add_argument('--fix_seq_file',  type=str,  required=False,
                    help='fix seq csv which contains file_path and fix res index, you can also provide bias column to do bias seqeunce design')
    parser.add_argument('--framework_seq',  type=str, nargs='+', required=False, default=[],
                    help='framework seq input to refine antibody, sequence defined in this will be fixed')
    parser.add_argument('--template_path', type=str, required=True,
                       help='Path to AF3 template.json file, it should be really careful to treat with')
    parser.add_argument('--template_for_eval', type=str, required=False, 
                       help='Path to AF3 template.json file only for eval')
    parser.add_argument('--extra_json_path', type=str, required=False, 
                       help='Path to AF3 template.json file only for cross model')       
    parser.add_argument('--HalluDesign_model', type=str,  required=True,
                       help='af3 or Protenix')
    parser.add_argument('--CoDP', action='store_true', default=False,
                    help='whether to use CoDP to validate sequence quality')
    parser.add_argument('--sm', type=str, nargs='+', required=False, default=[],
                    help='smiles input, it needs a good structure mapping smille')
    parser.add_argument('--mpnn',  type=str,  required=False,
                    help='which mpnn model do you choose proteinmpnn, ligandmpnn or ligandmpnn_plus_proteinmpnn (ligandmpnn for ligand binding part and proteinmpnn for protein part)')
    parser.add_argument('--mpnn_temperature', type=float, default=0.1,
                       help='mpnn temperature to use')
    parser.add_argument('--replace_MSA', action='store_true', default=False,
                       help='framework to use MSA, used in antibody or nanobody design')
    parser.add_argument('--ccd', type=str, nargs='+', required=False, default=[],
                    help='ccd input,it needs a good structure mapping ccd for AF3')
    parser.add_argument('--dna', type=str, nargs='+', required=False, default=[],
                    help='dna input')
    parser.add_argument('--rna', type=str, nargs='+', required=False, default=[],
                    help='rna input')
    parser.add_argument('--design_epoch_begin', type=int, required=False,  default=0,
                    help='in which cycles, multi seqs evaluation process will begin')
    parser.add_argument("--symmetry_residues",type=str,default="",
                    help="Add list of res for which residues need to be symmetric, e.g. 'A12,A13,A14|C2,C3|A5,B6'")
    parser.add_argument("--symmetry_chains",type=str,default="",
                    help="Add list of chains for which chains need to be symmetric, e.g. 'A,B'")
    parser.add_argument("--pocket_fix", action='store_true', default=False,
                    help="for repeat sample")
    parser.add_argument("--symmetry_segments",type=int,default=0,
                    help="Add symmetric constraints for Chain A, e.g. 5")
    parser.add_argument("--cyclic", type=int, default=0,
                    help="for cyclic peptide design, only for peptide A")
    parser.add_argument("--ptm",  type=str, nargs='+', required=False, default=[],
                    help="for ptm design")
    parser.add_argument("--random_init", action='store_true', default=False,
                    help="for pure noise generation")
    parser.add_argument("--random_init_chain_spec", type=str, default="",
                    help='No-PDB random-init protein chain length spec, e.g. "A:20" or "A:20-40"')
    parser.add_argument("--num_designs", type=int, default=1,
                    help="number of no-PDB random-init designs to generate")
    parser.add_argument("--seed", type=int, default=123,
                    help="random seed used for no-PDB random-init sequences and lengths")
    parser.add_argument("--enzyme_design", action='store_true', default=False,
                    help="for enzyme design")
    return parser.parse_args()


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _template_chain_records(template_path, halludesign_model):
    with open(template_path, "r") as handle:
        template = json.load(handle)

    records = []
    if halludesign_model == "af3":
        sequences = template.get("sequences", [])
        for sequence_index, item in enumerate(sequences):
            if "protein" in item:
                for chain_id in _as_list(item["protein"].get("id", [])):
                    records.append((str(chain_id), "protein", sequence_index, len(_as_list(item["protein"].get("id", [])))))
            elif "ligand" in item:
                for chain_id in _as_list(item["ligand"].get("id", [])):
                    records.append((str(chain_id), "ligand", sequence_index, len(_as_list(item["ligand"].get("id", [])))))
            elif "dna" in item:
                for chain_id in _as_list(item["dna"].get("id", [])):
                    records.append((str(chain_id), "dna", sequence_index, len(_as_list(item["dna"].get("id", [])))))
            elif "rna" in item:
                for chain_id in _as_list(item["rna"].get("id", [])):
                    records.append((str(chain_id), "rna", sequence_index, len(_as_list(item["rna"].get("id", [])))))
    else:
        chain_labels = iter(string.ascii_uppercase)
        sequences = template[0].get("sequences", [])
        for sequence_index, item in enumerate(sequences):
            if "proteinChain" in item:
                chain_type, count = "protein", int(item["proteinChain"].get("count", 1))
            elif "ligand" in item:
                chain_type, count = "ligand", int(item["ligand"].get("count", 1))
            elif "dnaSequence" in item:
                chain_type, count = "dna", int(item["dnaSequence"].get("count", 1))
            elif "rnaSequence" in item:
                chain_type, count = "rna", int(item["rnaSequence"].get("count", 1))
            else:
                continue
            for _ in range(count):
                records.append((next(chain_labels), chain_type, sequence_index, count))
    return records


def _parse_random_init_chain_spec(spec, template_path, halludesign_model):
    records = _template_chain_records(template_path, halludesign_model)
    record_by_chain = {chain_id.upper(): (chain_type, sequence_index, count) for chain_id, chain_type, sequence_index, count in records}
    if not record_by_chain:
        raise ValueError("No chains were found in the template JSON.")

    ranges = {}
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f'Invalid --random_init_chain_spec item "{item}". Use A:20 or A:20-40.')
        chain_id, length_spec = item.split(":", 1)
        chain_id = chain_id.strip().upper()
        length_spec = length_spec.strip()
        if chain_id not in record_by_chain:
            raise ValueError(f"Chain {chain_id} is not present in the template JSON.")
        chain_type, _, count = record_by_chain[chain_id]
        if chain_type != "protein":
            raise ValueError(f"Chain {chain_id} is a {chain_type} chain. Random-init length specs only support protein chains.")
        if count != 1:
            raise ValueError(
                f"Chain {chain_id} belongs to a template sequence block with count/id length {count}. "
                "Split that block into single-chain entries before changing random-init lengths."
            )
        if "-" in length_spec:
            low_text, high_text = length_spec.split("-", 1)
            low, high = int(low_text), int(high_text)
        else:
            low = high = int(length_spec)
        if low <= 0 or high <= 0 or low > high:
            raise ValueError(f"Invalid length range for chain {chain_id}: {length_spec}")
        ranges[chain_id] = (low, high)
    if not ranges:
        raise ValueError("--random_init_chain_spec is empty.")
    return ranges


def _make_random_init_sequences(length_ranges, seed):
    rng = random.Random(seed)
    sequences = {}
    for chain_id, (low, high) in length_ranges.items():
        length = rng.randint(low, high)
        sequences[chain_id] = "".join(rng.choice(PROTEIN_ALPHABET) for _ in range(length))
    return sequences


def _fix_seq_keys(value):
    if not value:
        return set()
    base = os.path.basename(str(value))
    stem = os.path.splitext(base)[0]
    return {str(value), base, stem}


def main():
    args = parse_arguments()
    args.output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    args.template_path = os.path.abspath(os.path.expanduser(args.template_path))
    if args.template_for_eval:
        args.template_for_eval = os.path.abspath(os.path.expanduser(args.template_for_eval))
    if args.extra_json_path:
        args.extra_json_path = os.path.abspath(os.path.expanduser(args.extra_json_path))
    if args.fix_seq_file:
        args.fix_seq_file = os.path.abspath(os.path.expanduser(args.fix_seq_file))
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.template_path):
        raise FileNotFoundError(f"Template file {args.template_path} not found!")
    
    if args.pdb_list and args.input_file:
        raise ValueError("Cannot specify both --pdb_list and --input_file at the same time")

    if args.sm and args.ccd:
        raise ValueError("Cannot specify both --sm and --ccd at the same time, only allow one")
    
    if args.symmetry_residues and args.symmetry_chains:
        raise ValueError("Cannot specify both --symmetry_residues and --symmetry_chains at the same time, only allow one")

    if args.random_init_chain_spec and (args.pdb_list or args.input_file):
        raise ValueError("Do not combine --random_init_chain_spec with --pdb_list or --input_file")

    if args.num_designs < 1:
        raise ValueError("--num_designs must be at least 1")

    random_init_length_ranges = None
    if args.random_init_chain_spec:
        args.random_init = True
        random_init_length_ranges = _parse_random_init_chain_spec(
            args.random_init_chain_spec,
            args.template_path,
            args.HalluDesign_model,
        )
        pdb_files = [None for _ in range(args.num_designs)]
    elif args.pdb_list:
        if args.num_designs != 1:
            raise ValueError("--num_designs is only supported with --random_init_chain_spec")
        with open(args.pdb_list, 'r') as f:
            pdb_files = [os.path.abspath(os.path.expanduser(line.strip())) for line in f if line.strip()]
    elif args.input_file:
        if args.num_designs != 1:
            raise ValueError("--num_designs is only supported with --random_init_chain_spec")
        pdb_files = [os.path.abspath(os.path.expanduser(args.input_file))]
    else:
        raise ValueError("You must specify --pdb_list, --input_file, or --random_init_chain_spec")

    if not args.mpnn:
        args.mpnn = "ligand_mpnn" if (args.sm or args.dna or args.rna or args.ccd) else "protein_mpnn"

    if args.random_init_chain_spec and args.enzyme_design:
        raise ValueError("--enzyme_design currently requires an input PDB filename with Hpos metadata.")
        
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
        "model_name": args.mpnn, 
        "num_seqs": 1
        }
    print(f"{args.mpnn} will be use in sequnce design")
    if args.fix_res_index:
        fixed_residues_base = args.fix_res_index.split()
    else:
        fixed_residues_base = []
    evaluator = None
    if args.CoDP and args.HalluDesign_model != "af3":
        checkpoints_to_run = "./CoDP/ckpt/epoch_1_without_esm2.pth"
        esm_name = "facebook/esm2_t33_650M_UR50D"
        #! remain to do!
        evaluator = CoDP(checkpoints_to_run,esm_name)
    elif args.CoDP and args.HalluDesign_model == "af3":
        evaluator = "subprocess_CoDP"
    if args.fix_chain_index:
        fixed_chains = args.fix_chain_index.split()
    else:
        fixed_chains = []

    if args.HalluDesign_model  == "af3":
        from af3_model import AF3DesignerPack
        Designer_model = AF3DesignerPack(jax_compilation_dir=os.path.join(args.output_dir,"jax_compilation_cache_dir"))
        if args.template_for_eval is not None:
            template_path_for_eval = args.template_for_eval
        else:
            template_path_for_eval = args.template_path
        protein_chains, ligand_chains, dna_chains, rna_chains, chain_types =  count_chain_based_on_json_af3(template_path_for_eval)
        metrics = generate_metrics(protein_chains,ligand_chains, dna_chains, rna_chains,chain_types)
    if args.HalluDesign_model  == "protenix" or args.HalluDesign_model == "cross_model":
        #sys.path.insert(0,os.path.join(current_dir,"Protenix"))
        from runner.inference import ProtenixInferrer
        os.environ["LAYERNORM_TYPE"] = "fast_layernorm"
        os.environ["USE_DEEPSPEED_EVO_ATTENTION"] = "true"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.environ["CUTLASS_PATH"] = os.path.join(script_dir, "cutlass")
        static_configs = {
        "model.N_cycle": 10,
        "sample_diffusion.N_sample": 5,
        "sample_diffusion.N_step": 200, # Example value
        "use_esm": True,
        "use_msa": False,
        "need_atom_confidence": True, 
        "sorted_by_ranking_score": True,
        # Add any other global or model-specific configs here
        # "load_checkpoint_path": "/path/to/your/model_checkpoint.pth", # Ensure this is correct
        # "need_atom_confidence": True, # Or False, depending on your needs
        #  # Or False
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
    for job_index, pdb_file in enumerate(pdb_files):
        random_init_sequences = None
        random_init_file_tag = None
        if pdb_file is None:
            random_init_file_tag = f"random_init_{job_index + 1:03d}"
            random_init_sequences = _make_random_init_sequences(
                random_init_length_ranges,
                args.seed + job_index,
            )
            print(f"\nProcessing {random_init_file_tag} with random-init sequences: "
                  f"{ {chain: len(seq) for chain, seq in random_init_sequences.items()} }")
        else:
            print(f"\nProcessing {pdb_file}...")
        current_input = pdb_file
        chain_number_list_cdr = []
        bais_per_residues = None
        fixed_residues = copy.deepcopy(fixed_residues_base)
        if  args.fix_seq_file:
            df_fix = pd.read_csv(args.fix_seq_file)
            fix_match_keys = _fix_seq_keys(random_init_file_tag if pdb_file is None else pdb_file)
            if "fix_res" in df_fix.columns:
                df_fix_file = df_fix[df_fix['file_path'].apply(lambda x: bool(_fix_seq_keys(x) & fix_match_keys))]["fix_res"]
            
                if not df_fix_file.empty:
                    fixed_residues = str(df_fix_file.values[0]).split()
            if "bias" in df_fix.columns:
                bias = df_fix[df_fix["file_path"].apply(lambda x: bool(_fix_seq_keys(x) & fix_match_keys))]["bias"]
    
                if not bias.empty:
                    bais_per_residues = ast.literal_eval(bias.values[0])
        if args.enzyme_design:         
            fixed_residues = parse_his_positions_from_pdb_filename(pdb_file)
            print(fixed_residues)    
        for cycle in range(args.num_recycles):
            print(f"  Starting cycle {cycle+1}")
            try:
                is_last_cycle = (cycle == args.num_recycles - 1)
                design_begin = False
                mpnn_config_dict['num_seqs'] = 1
                if cycle >= args.design_epoch_begin:
                    mpnn_config_dict['num_seqs'] = args.num_seqs
                    design_begin = True
                print(f"begin multi-batch evaluation {design_begin}")
                metrics = copy.deepcopy(metrics_new)
                if args.HalluDesign_model  == "af3":
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
                    args.enzyme_design,
                    run_af3=not is_last_cycle,  #  AF3 not run in last cycle
                    random_init_sequences=random_init_sequences,
                    random_init_file_tag=random_init_file_tag
                )
                elif args.HalluDesign_model  == "protenix":
                    metrics, next_input, chain_number_list_cdr = protenix_op_protenix_eval(
                    pdb_file=current_input,
                    cycle=cycle,
                    output_dir=args.output_dir,
                    template_path=args.template_path,
                    template_for_eval=args.template_for_eval,
                    mpnn_model=mpnn_model,
                    mpnn_config_dict=mpnn_config_dict,
                    Designer_model=Designer_model,
                    ref_time_steps=args.ref_time_steps,
                    chain_types=chain_types,
                    fixed_chains=fixed_chains,
                    fixed_residues=fixed_residues,
                    bais_per_residues=bais_per_residues,
                    metrics=metrics,
                    symmetry_residues=args.symmetry_residues,
                    symmetry_chains=args.symmetry_chains,
                    sm=args.sm,
                    dna=args.dna,
                    rna=args.rna,
                    cdr=args.cdr,
                    framework_seq=args.framework_seq,
                    evaluator=evaluator,
                    design_begin=design_begin,
                    chain_number_list_cdr=chain_number_list_cdr,
                    cyclic=args.cyclic,
                    random_init=args.random_init,
                    run_af3=not is_last_cycle,
                    random_init_sequences=random_init_sequences,
                    random_init_file_tag=random_init_file_tag
                )
                elif args.HalluDesign_model  == "cross_model":
                    metrics, next_input, chain_number_list_cdr = cross_model_op_protenix_eval(
                    pdb_file=current_input,
                    cycle=cycle,
                    output_dir=args.output_dir,
                    template_path=args.template_path,
                    template_for_eval=args.template_for_eval,
                    extra_json_path=args.extra_json_path,
                    mpnn_model=mpnn_model,
                    mpnn_config_dict=mpnn_config_dict,
                    Designer_model=Designer_model,
                    ref_time_steps=args.ref_time_steps,
                    chain_types=chain_types,
                    fixed_chains=fixed_chains,
                    fixed_residues=fixed_residues,
                    bais_per_residues=bais_per_residues,
                    metrics=metrics,
                    symmetry_residues=args.symmetry_residues,
                    symmetry_chains=args.symmetry_chains,
                    symmetry_segments=args.symmetry_segments,
                    sm=args.sm,
                    dna=args.dna,
                    rna=args.rna,
                    cdr=args.cdr,
                    framework_seq=args.framework_seq,
                    evaluator=evaluator,
                    design_begin=design_begin,
                    chain_number_list_cdr=chain_number_list_cdr,
                    cyclic=args.cyclic,
                    ptm=args.ptm,
                    random_init=args.random_init,
                    enzyme_design=args.enzyme_design,
                    run_af3=not is_last_cycle,
                    random_init_sequences=random_init_sequences,
                    random_init_file_tag=random_init_file_tag
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
