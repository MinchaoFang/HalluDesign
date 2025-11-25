import torch
import os
import logging
import pandas as pd
import numpy as np
import argparse
import sys
import time
import dataclasses
import re
from functools import partial
sys.path.append("/storage/caolab/wangct/Project/Proteus_flow_matching")
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from omegaconf import OmegaConf
from data import utils as du
from data import residue_constants as rc
from data import data_transforms
from data import protein
from data.parsers import from_pdb_string
from analysis import metrics as am
from ProteinMPNN.protein_mpnn_utils import model_init
from ProteinMPNN.protein_mpnn_pyrosetta import mpnn_design
import esm.esmfold.v1.pretrained
import esm.esmfold.v1.esmfold
from colabdesign.af.model import mk_af_model
from colabdesign import clear_mem

import pyrosetta
pyrosetta.init('-ignore_unrecognized_res -ignore_zero_occupancy -mute all -corrections::beta_nov16 true')
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from Bio.PDB.Polypeptide import aa3
logging.basicConfig(level=logging.INFO,
    # define this things every time set basic config
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ])

from Bio.PDB import MMCIFParser, PDBParser, Superimposer

import numpy as np

def calculate_rmsd_for_residues(ciffile: str, pdbfile: str, residues: list) -> float:
    """
    Calculate the all-atom RMSD of specified residues between two structure files.

    Parameters:
        ciffile (str): Path to the CIF file.
        pdbfile (str): Path to the PDB file.
        residues (list[str]): List of residue identifiers to compare,
                              formatted as ["A1", "B5"], where the letter
                              represents the chain ID and the number the residue index.

    Returns:
        float: The computed RMSD value.
    """
    try:
        cif_parser = MMCIFParser(QUIET=True)
        cif_structure = cif_parser.get_structure("cif_structure", ciffile)


        pdb_parser = PDBParser(QUIET=True)
        pdb_structure = pdb_parser.get_structure("pdb_structure", pdbfile)


        cif_atoms = []
        pdb_atoms = []

        for model in cif_structure:
            for chain in model:
                for residue in chain:
                    res_id = str(residue.id[1])
                    if residue.id[2].strip(): 

                        res_id += residue.id[2]
                    residue_code = f"{chain.id}{res_id}"
                    
                    if residue_code in residues:
                        cif_atoms.extend(residue.get_atoms())

        for model in pdb_structure:
            for chain in model:
                for residue in chain:
                    res_id = str(residue.id[1])
                    if residue.id[2].strip(): 

                        res_id += residue.id[2]
                    residue_code = f"{chain.id}{res_id}"
                    
                    if residue_code in residues:
                        pdb_atoms.extend(residue.get_atoms())


        if len(cif_atoms) != len(pdb_atoms):
            raise ValueError("Error: Atom count mismatch between residues. Cannot calculate RMSD.")

        # cal RMSD

        super_imposer = Superimposer()
        super_imposer.set_atoms(cif_atoms, pdb_atoms)
        super_imposer.apply(cif_structure.get_atoms())

        return super_imposer.rms

    except Exception as e:
        print(f"Error: {str(e)}")
        return None


from Bio.PDB import PDBParser, MMCIFParser, NeighborSearch

def find_pocket_residues_based_on_distance(pdbfile: str, cutoff=5.0) -> list:
    """
    Identify all protein residues that are within a specified distance (cutoff)
    from any non-protein residue in a given PDB file.

    Parameters:
        pdbfile (str): Path to the PDB file.
        cutoff (float): Distance threshold in Ångströms.

    Returns:
        list[str]: A list of residue identifiers within the cutoff distance,
                   formatted as ["A1", "B5"], where the letter denotes the
                   chain ID and the number denotes the residue index.
    """
    try:
        if pdbfile.endswith(".cif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        structure = parser.get_structure("target", pdbfile)
        
        sm_atoms = []    

        protein_atoms = []  

        nearby_residues = set()  

        STANDARD_AMINO_ACIDS = set([aa.upper() for aa in aa3])


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

        if not sm_atoms or not protein_atoms:
            print("Error: no ligand or protein atoms")
            return []

        ns = NeighborSearch(protein_atoms)

        for atom in sm_atoms:
            close_atoms = ns.search(atom.coord, cutoff)
            for nearby_atom in close_atoms:
                residue = nearby_atom.get_parent()
                chain = residue.get_parent()
                
                res_id = str(residue.id[1])
                if residue.id[2].strip():  
                    res_id += residue.id[2]
                
                residue_code = f"{chain.id}{res_id}"
                nearby_residues.add(residue_code)

        def sort_key(code):
            chain_part = code[0]
            num_part = ''.join(filter(str.isdigit, code[1:])) or '0'
            return (chain_part, int(num_part))

        return sorted(nearby_residues, key=sort_key)

    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def run_folding_and_evaluation(model, sequence, scaffold_model, ref_feats=None, af2_setting=None, template_chains=None):
    """Run ESMFold or AlphaFold2 on sequence."""
    evaluated_results = []
    pred_prots = []
    
    if isinstance(model, esm.esmfold.v1.esmfold.ESMFold):
        with torch.no_grad():
            output = model.infer(sequence)
        output = {key: value.cpu() for key, value in output.items()}
        output['final_atom_positions'] = data_transforms.atom14_to_atom37(output["positions"][-1], output)
        output = {k: v.numpy()[0] for k, v in output.items() if k in ["aatype", "plddt","final_atom_positions","atom37_atom_exists", "residue_index", "chain_index", "aligned_confidence_probs"]}
        residue_mask = output["atom37_atom_exists"][:,1] == 1
        output = {key: (value[residue_mask] if key != 'aligned_confidence_probs' else value[residue_mask,:][:,residue_mask]) for key, value in output.items()}
        pred_prot = protein.Protein(
            aatype=output["aatype"],
            atom_positions=output['final_atom_positions'],
            atom_mask=output["atom37_atom_exists"],
            residue_index=output["residue_index"] + 1,
            b_factors=output["plddt"],
            chain_index=output["chain_index"] if "chain_index" in output else None,
        )
        pred_prots.append(pred_prot)
        pae = ((output["aligned_confidence_probs"] * np.arange(64)).mean(-1) * 31).mean()
        confidence_metrics = {
            'prediction_model': 'esmfold',
            'pae': pae,
            'plddt': output['plddt'][:,1].mean(),
        }
        self_consistency_metrics = am.self_consistency_eval(scaffold_model, pred_prot, ref_feats)
        evaluated_results.append({**confidence_metrics, **self_consistency_metrics})
        del output
    else:  # AlphaFold2
        if model.protocol == "hallucination":
            sequence = re.sub("[^A-Z]", "", sequence.upper())
            model.prep_inputs(length=len(sequence))
            model.set_seq(sequence)
        if model.protocol == "fixbb":
            sequence = re.sub("[^A-Z]", "", sequence.upper())
            model.prep_inputs(protein.to_pdb(scaffold_model), chain=",".join(np.unique(scaffold_model.chain_index).tolist()))
            model.restart(rm_aa="C,M")
            model.set_seq(sequence)
        if model.protocol == "binder":
            target_chain = ",".join([ch for ch in np.unique(scaffold_model.chain_index).tolist() if ch in template_chains])
            print(f"Target chain: {target_chain}, binder length : {len(sequence.split(':')[0])}")
            binder_chain = ",".join([ch for ch in np.unique(scaffold_model.chain_index).tolist() if ch not in template_chains])
            model.prep_inputs(
                protein.to_pdb(scaffold_model), 
                chain=target_chain,
                # binder_len=len(sequence.split(':')[0]),
                binder_chain=binder_chain,
                rm_target_seq=True
            )
        for model_num in models:
            if model.protocol == "binder":
                binder_seq = sequence.split(':')[0]
                model.predict(seq=binder_seq, models=[model_num], num_recycles=af2_setting["num_recycles"], verbose=False)
            else:
                model.predict(models=[model_num], num_recycles=af2_setting["num_recycles"], verbose=False)
            pred_prot = from_pdb_string(model.save_pdb())
            confidence_metrics = {
                'prediction_model': f'alphafold_{af2_setting["prefix"]}_{model_num+1}',
                'plddt': round(model.aux["log"]["plddt"], 2),
                'pae': round(model.aux["log"]["pae"], 2),
                'ptm': round(model.aux["log"]["ptm"], 2),
            }
            if np.unique(pred_prot.chain_index).shape[0] > 1:
                confidence_metrics.update({
                    'i_ptm': round(model.aux["log"]["i_ptm"], 2),
                })
            self_consistency_metrics = am.self_consistency_eval(scaffold_model, pred_prot, ref_feats)
            evaluated_results.append({**confidence_metrics, **self_consistency_metrics})
            pred_prots.append(pred_prot)
    
    return evaluated_results, pred_prots



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-metadata", type=str, default=None, help="csv file contains the information of scaffold")
    parser.add_argument("-shuffle", action="store_true", help='Whether to shuffle the metadata before running')
    parser.add_argument("-start_idx", type=int, default=0, help="The start index of metadata to run")
    parser.add_argument("-end_idx", type=int, default=None, help="The end index of metadata to run (default: None, run till end)")
    parser.add_argument("-seqs_per_struct", type=int, default=8, help="The number of sequences to generate for each structure (default: 1)")
    parser.add_argument("-temperature", type=float, default=0.1, help='The sampling temperature to use when running ProteinMPNN (default: 0.1)')
    parser.add_argument("-ca_only", action="store_true", help='Whether to use only CA atoms in the ProteinMPNN model (default: False)')
    parser.add_argument("-initial_guess", action="store_true")
    parser.add_argument("-initial_atom_pos", action="store_true")
    parser.add_argument("-alphafold", action="store_true", help='use af2 as prediction model')
    parser.add_argument("-esmfold", action="store_true", help='use esmfold as prediction model')
    parser.add_argument("-models", nargs='+', type=int, default=[0,1,2,3,4])
    parser.add_argument("-multimer", action="store_true", help='Whether to predict multimeric structures instead of first chain')
    parser.add_argument("-pocket", action="store_true", help='Whether to align small pockets sidechain')
    parser.add_argument("-template_chains", nargs='+', type=str, default=None, help="chains predicted with template")
    parser.add_argument("-interface_analyzer", action="store_true")
    parser.add_argument("-output_dir", type=str, default='./', help='The directory of prediction result')
    parser.add_argument("-suffix", type=str, default=None, help='suffix to append to the output files')

    args = parser.parse_args(sys.argv[1:])

    af2_setting = {
        "models": args.models if args.template_chains is None else [ch for ch in args.models if ch in [0,1]],
        "num_recycles": 3,
        'prefix': 'monomer',
        'params_dir': f'{os.path.dirname(__file__)}/../weights/alphafold/'
    }

    metadata_df = pd.read_csv(args.metadata)
    metadata_df = metadata_df.dropna(subset=['packed_path'])
    #metadata_df = metadata_df.drop('mpnn_sequence', axis=1)
    if args.shuffle:
        metadata_df = metadata_df.sample(frac=1, random_state=42).reset_index(drop=True)
    if args.end_idx is None or args.end_idx > len(metadata_df):
        args.end_idx = len(metadata_df)
    metadata_df = metadata_df.iloc[args.start_idx:args.end_idx]
    metadata_df = metadata_df.sort_values('length', ascending=True) if 'length' in metadata_df.columns else metadata_df
    print(f"Processing metadata rows from {args.start_idx} to {args.end_idx} (total {len(metadata_df)} rows)")
    
    mpnn_model = None
    if 'mpnn_sequence' not in metadata_df.columns:
        mpnn_config_dict = {
            "ca_only": args.ca_only,
            "model_name": "v_48_020",
            'backbone_noise': 0.00,
            'temperature': args.temperature,
            'num_seqs': args.seqs_per_struct,
        }
        mpnn_config_dict = OmegaConf.create(mpnn_config_dict)
        mpnn_model = model_init(mpnn_config_dict, device='cuda')
    else:
        print("Sequence exists in metadata, skipping sequence design")
        
    if args.alphafold:
        clear_mem()
        if args.template_chains is None:
            prediction_model = mk_af_model(
                protocol="hallucination" if not args.initial_guess and not args.initial_atom_pos else "fixbb", 
                use_templates=False, initial_guess=args.initial_guess, 
                use_initial_atom_pos=args.initial_atom_pos, num_recycles=3, 
                data_dir=af2_setting['params_dir']
            )
        else:
            prediction_model = mk_af_model(
                protocol="binder", 
                initial_guess=args.initial_guess, 
                use_initial_atom_pos=args.initial_atom_pos, num_recycles=3, 
                data_dir=af2_setting['params_dir'],
            )
        prediction_dir = os.path.join(args.output_dir, 'alphafold')
        os.makedirs(prediction_dir, exist_ok=True)
    elif args.esmfold:
        prediction_model = esm.pretrained.esmfold_v1()
        prediction_model = prediction_model.eval().cuda()
        prediction_model.set_chunk_size(512)
        prediction_dir = os.path.join(args.output_dir, 'esmfold')
        os.makedirs(prediction_dir, exist_ok=True)
    else:
        prediction_model = None
        

    for scaffold_path, scaffold_group in metadata_df.groupby('packed_path'):
        start_time = time.time()
        scaffold_group = scaffold_group.reset_index(drop=True)
        row = scaffold_group.iloc[0]
        basename = os.path.splitext(os.path.basename(scaffold_path))[0]
        try:
            scaffold_prot = from_pdb_string(open(scaffold_path).read(), 'A' if not args.multimer else ["A","B"])
        except:
            continue
        length = scaffold_prot.atom_positions.shape[0]
        
        reference_feats = None
        if 'reference_path' in row:
            reference_prot = from_pdb_string(open(row['reference_path']).read())
            reference_feats = dataclasses.asdict(reference_prot)
            reference_feats = du.init_feat(
                contigs=row['contigs'],
                ref_feats=reference_feats,
                mask_aatype=row['mask_aatype'].split(',') if 'mask_aatype' in metadata_df.columns and pd.notna(row['mask_aatype']) else None,
            )
            ref_mask = du.to_numpy(reference_feats['ref_mask'])[0]
            reference_feats = {k: du.to_numpy(v)[0, ~ref_mask] for k, v in reference_feats.items()}
        
        if 'mpnn_sequence' not in scaffold_group.columns:
            assert scaffold_group.shape[0] == 1, "multiple scaffolds exists"
            
            mpnn_seqs, mpnn_scores = mpnn_design(
                config=mpnn_config_dict,
                protein_path=scaffold_path,
                model=mpnn_model,
                design_chains=['A']
            )
            
            struct2seq_results = [{'mpnn_sequence': seq[:length], 'sequence_score': score} for seq, score in zip(mpnn_seqs, mpnn_scores)]
            struct2seq_df = pd.DataFrame(struct2seq_results)
            for column, value in scaffold_group.iloc[0].to_dict().items():
                struct2seq_df[column] = value  
        else:
            struct2seq_df = scaffold_group
        struct2seq_df['scaffold_path'] = scaffold_path
  
        struct2seq_df = struct2seq_df.drop_duplicates(subset=['mpnn_sequence'], keep='first')
        mpnn_seqs = struct2seq_df['mpnn_sequence'].values
        
        if prediction_model is None:
            metadata_path = os.path.join(args.output_dir, f"mpnn.csv")
            struct2seq_df.to_csv(metadata_path, mode='a', header=not os.path.exists(metadata_path), index=False)
            continue
  
        self_consistency_results = []
        for i in range(len(mpnn_seqs)):
            chain_separator = np.where(scaffold_prot.chain_index[:-1] != scaffold_prot.chain_index[1:])[0] + 1
            chain_nums = chain_separator.shape[0] + 1
            first_chain_length = chain_separator[0] if chain_nums > 1 else length
            
            if isinstance(prediction_model, esm.esmfold.v1.esmfold.ESMFold) and os.path.exists(os.path.join(prediction_dir, f"{basename}_mpnn_{i}_esmfold{('_'+args.suffix) if args.suffix is not None else '' }.pdb")):
                skip = True
            else:
                models_to_run = []
                for model_num in af2_setting['models']:
                    model_path = os.path.join(prediction_dir, f"{basename}_mpnn_{i}_alphafold_{af2_setting['prefix']}_{model_num+1}{('_'+args.suffix) if args.suffix is not None else '' }.pdb")
                    if not os.path.exists(model_path):
                        models_to_run.append(model_num)
                skip = len(models_to_run) == 0
                models = models_to_run
            
            if skip:
                logging.info(f'Skipping scaffold {scaffold_path}')
                continue
            
            # correctly handle the case of single chain and multi chain
            if args.multimer and chain_nums > 1:
                if length > len(mpnn_seqs[i]) and ':' not in mpnn_seqs[i]:
                    assert len(mpnn_seqs[i]) == first_chain_length, f"First chain length mismatch: {len(mpnn_seqs[i])} != {length}"
                    # retrieve additional chain sequences from the scaffold input
                    sequence = mpnn_seqs[i] + "".join([rc.restypes[i] for i in scaffold_prot.aatype.tolist()])[length:]
                    assert len(sequence) == length, f"Length mismatch: {len(sequence)} != {length}"
                    sequence = list(sequence)
                    for pos in reversed(chain_separator):
                        sequence.insert(pos, ':')
                    sequence = ''.join(sequence)
                elif ':' in mpnn_seqs[i]:
                    assert len(mpnn_seqs[i].replace(':','')) == length, f"Complex Length mismatch: {len(mpnn_seqs[i].replace(':',''))} != {length}"
                    sequence = mpnn_seqs[i]
            else:
                sequence = mpnn_seqs[i].split(':')[0] if ':' in mpnn_seqs[i] else mpnn_seqs[i]
                assert len(sequence) == first_chain_length, f"Length mismatch: {len(sequence)} != {first_chain_length}"
            #print(sequence)
            evaluated_results, pred_prots = run_folding_and_evaluation(prediction_model, sequence, scaffold_prot, reference_feats, af2_setting, template_chains=args.template_chains)
            
            if isinstance(prediction_model, esm.esmfold.v1.esmfold.ESMFold):
                # ESMFold case
                result = evaluated_results[0]  # Only one result for ESMFold
                pred_prot = pred_prots[0]
                fold_path = os.path.abspath(os.path.join(prediction_dir, f"{basename}_mpnn_{i}_esmfold{('_'+args.suffix) if args.suffix is not None else '' }.pdb"))
                with open(fold_path, 'w') as f:
                    f.write(protein.to_pdb(pred_prot))  # pred_prots is a single protein for ESMFold
                result["esmfold_path"] = fold_path
                result['mpnn_sequence'] = mpnn_seqs[i]
                self_consistency_results.append(result)
            else:
                # AlphaFold2 case
                for j, (result, pred_prot) in enumerate(zip(evaluated_results, pred_prots)):
                    fold_path = os.path.abspath(os.path.join(prediction_dir, f"{basename}_mpnn_{i}_{result['prediction_model']}{('_'+args.suffix) if args.suffix is not None else '' }.pdb"))
                    with open(fold_path, 'w') as f:
                        f.write(protein.to_pdb(pred_prot))
                    result["alphafold_path"] = fold_path
                    result['mpnn_sequence'] = mpnn_seqs[i]
                    if args.pocket:
                        pockets_res = find_pocket_residues_based_on_distance(row["eval_path"])
                        result["pocket_res_len"] = len(pockets_res)
                        print(len(pockets_res))
                        rmsd_value = calculate_rmsd_for_residues(row["eval_path"],fold_path, pockets_res)
                        print(f"RMSD: {rmsd_value} Å")
                        result["pocket_all_atom_rmsd"] = rmsd_value
                    self_consistency_results.append(result)
        
        if len(self_consistency_results) == 0:
            continue
        
        predict_df = pd.DataFrame(self_consistency_results)
        predict_df = pd.merge(
            struct2seq_df.drop(columns=[col for col in struct2seq_df.columns if col in predict_df.columns and col != 'mpnn_sequence']), 
            predict_df, on='mpnn_sequence'
        )
        logging.info(f'Run self-consistency evaluation for scaffold {row["packed_path"]} with {len(predict_df.mpnn_sequence.unique())} seqs in {time.time()-start_time:.2f}s')
        logging.info(predict_df)

        if 'mpnn_sequence' not in metadata_df.columns:
            metadata_path = os.path.join(args.output_dir, "self_consistency.csv")
        else:
            metadata_path = os.path.join(args.output_dir, f"{'af2' if args.alphafold else 'esmfold'}{('_'+args.suffix) if args.suffix else ''}_prediction.csv")
        predict_df.to_csv(metadata_path, mode='a', header=not os.path.exists(metadata_path), index=False)