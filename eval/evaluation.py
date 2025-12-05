import os
import subprocess
from typing import Dict, List, Tuple
import sys
from scripts.input_pkl_preprocess import process_single_file
from eval.eval_utility import get_random_seeds,PeptideSynthesizer,get_gpu_memory
from data.op_utility import most_central_residue_resseq
import copy
import os
import warnings
#import scripts.pose_sequence 
from Bio import PDB
from data.utility import *
warnings.filterwarnings("ignore", category=PDB.PDBExceptions.PDBConstructionWarning)
import pandas as pd



import torch

import numpy as np

try:
    import biotite.structure.io as strucio
    from biotite.structure import AtomArray
    import biotite.structure as struc
except:
    print("no biotitle")
def read_pdb_to_atom_array(pdb_path: str):
    """
    Reads structure data from a PDB file and returns an AtomArray.

    Args:
        pdb_path (str): Path to the PDB file.

    Returns:
        AtomArray: The array of atoms from the PDB file.
    """
    try:
        # Automatically detect PDB format based on file extension

        atom_array = strucio.load_structure(pdb_path)  # Load PDB file

        return atom_array

    except Exception as e:
        print(f"Error reading PDB file: {e}")
        raise

def extract_coordinates(atom_array, as_tensor: bool = True):
    """
    Extracts atomic coordinates and saves them as a Tensor or NumPy array.

    Args:
        atom_array (AtomArray): The AtomArray parsed from the PDB file.
        as_tensor (bool): Whether to convert coordinates to a PyTorch Tensor. 
                          If False, a NumPy array is returned.

    Returns:
        torch.Tensor or np.ndarray: The extracted atomic coordinates.
    """
    try:
        coordinates = atom_array.coord # Extract atomic coordinates (NumPy array)
        if as_tensor:
            return torch.tensor(coordinates, dtype=torch.float32)  # Convert to PyTorch Tensor

        return coordinates  # Return NumPy array

    except AttributeError as e:
        print(f"Error extracting coordinates: {e}")
        raise

from scipy.stats import multivariate_normal

def generate_gaussian_from_residues(
    atom_array ,
    chain_id: str,
    res_indices: list[int],
    num_samples: int,
    std_dev: float = 1.0, # You can adjust this to control the spread of the Gaussian
    as_tensor: bool = True
):
    """
    Generates an array of 3D points sampled from a Gaussian distribution
    centered around the average coordinates of specified residues and their
    immediate neighbors (+1 and -1).

    Args:
        atom_array (AtomArray): The AtomArray loaded from a PDB file.
        chain_id (str): The ID of the chain to target (e.g., 'A', 'B').
        res_indices (list[int]): A list of residue indices within the specified chain.
        num_samples (int): The number of points to sample from the Gaussian distribution.
        std_dev (float): The standard deviation for the Gaussian distribution.
                         Controls the spread of the generated points.
        as_tensor (bool): If True, returns a PyTorch tensor; otherwise, a NumPy array.

    Returns:
        torch.Tensor or np.ndarray: An array of shape (num_samples, 3)
                                    containing the sampled 3D coordinates.
    Raises:
        ValueError: If no atoms are found for the specified chain and residue indices.
    """
    all_target_coords = []

    # Iterate through each specified residue index
    for res_index in res_indices:
        # Include the current residue and its neighbors (+1 and -1)
        # Ensure we don't go below residue index 1
        neighbor_indices = [idx for idx in [res_index - 1, res_index, res_index + 1] if idx >= 1]

        # Select atoms belonging to the specified chain and the current set of residue indices
        # We use 'np.isin' for efficient checking of multiple residue IDs
        # and 'atom_array.filter_amino_acid()' to specifically target protein residues
        # and potentially remove water or heteroatoms if desired, though 'filter_residue_id' is more general
        
        # More robust filtering:
        # 1. Filter by chain ID
        # 2. Filter by residue ID within that chain
        chain_mask = (atom_array.chain_id == chain_id)
        res_mask = np.isin(atom_array.res_id, neighbor_indices)
        
        target_atoms = atom_array[chain_mask & res_mask]

        if target_atoms.array_length == 0:
            print(f"Warning: No atoms found for chain '{chain_id}' at residue indices {neighbor_indices}.")
            continue

        all_target_coords.append(target_atoms.coord)

    if not all_target_coords:
        raise ValueError(
            f"No atoms found for the specified chain '{chain_id}' and residue indices {res_indices}. "
            "Cannot generate Gaussian distribution."
        )

    # Concatenate all collected coordinates and calculate the mean
    combined_coords = np.concatenate(all_target_coords, axis=0)
    mean_coordinates = np.mean(combined_coords, axis=0)

    # Define the covariance matrix for the Gaussian distribution
    # We'll use an isotropic covariance matrix for a spherical Gaussian
    covariance_matrix = np.eye(3) * (std_dev ** 2)

    # Create a multivariate normal distribution
    gaussian_distribution = multivariate_normal(mean=mean_coordinates, cov=covariance_matrix)

    # Sample points from the distribution
    sampled_points = gaussian_distribution.rvs(size=num_samples)

    if as_tensor:
        return torch.tensor(sampled_points, dtype=torch.float32)
    else:
        return sampled_points


def self_consistency_protenix(scaffold_path,
                        AF3Designer_model,
                        output_dir,
                        template_path,
                        sm,  
                        ccd_LG,
                        ccd_codes,
                        dna,
                        rna,
                        ref_eval,
                        chain_types,
                        pocket_res,
                        fixed_chains,
                        plddt_good_indicates,
                        cyclic,
                        metrics,
                        random_init= False):
    metrics_to_tile = []
    seq_count = 0
    for metric in metrics:
        cyclic_prediction = cyclic
        packed = metric["packed_path"]
        seq = metric['mpnn_sequence'] 
        
        # Prepare input JSON for AF3
        with open(template_path, 'r') as f:
            template = json.load(f)
        input_json = template.copy()
        # Safely get the base name of the scaffold file, change extension to .fa
        scaffold_basename = os.path.splitext(os.path.basename(scaffold_path))[0]
        input_json[0]['name'] = f"{scaffold_basename}_{seq_count}"
        json_filename = f"{scaffold_basename}_{seq_count}.json"
        json_path = os.path.join(output_dir, json_filename)
        count = 0
        sm_count = 0
        rna_count = 0 
        dna_count = 0 
        protein_count = 0
        print(seq)
        chain_labels = string.ascii_uppercase[:10] 
        for chain in chain_types:
            if chain == "protein":
                if chain_labels[protein_count] not in fixed_chains:
                    input_json[0]['sequences'][count]['proteinChain']["sequence"] = seq.split(':')[protein_count]
                protein_count +=1
            if chain == 'ligand':
                input_json[0]['sequences'][count]["ligand"]["ligand"] = sm[sm_count]
                sm_count +=1
                if random_init:
                    centre_res = most_central_residue_resseq(metric["packed_path"])
                    input_json[0]['constraint']['contact'][0]["position1"] = centre_res
            if chain == 'dna':
                input_json[0]['sequences'][count]["dnaSequence"]["sequence"] = dna[sm_count]
                dna_count +=1
            if chain == 'rna':
                input_json[0]['sequences'][count]["rnaSequence"]["sequence"] = rna[sm_count]
                rna_count +=1
            count += 1
        if ccd_LG:
            input_json['userCCD']=ccd_LG
        # Write the new JSON file
        with open(json_path, 'w') as f:
            json.dump(input_json, f, indent=2)
            
        base_name = os.path.basename(scaffold_path)
       
        if ref_eval:
            print("normal AF3 batch diffusion plus ref guided diffusion")
            
            ref_out_dir = os.path.join(output_dir, "packed")
            # Run AF3
            pkl_path = os.path.join(ref_out_dir, f'{os.path.splitext(packed)[0]}.pkl')
            atom_array = read_pdb_to_atom_array(packed)
            pred_coordinates_tensor = extract_coordinates(atom_array, as_tensor=True)
            torch.save(pred_coordinates_tensor,pkl_path)
            prediction=AF3Designer_model.predict(
            input_json_path=json_path,
            dump_dir=output_dir,
            seed=123,
            #input_atom_array_path="/storage/caolongxingLab/fangminchao/software/Protenix/coord_pdb.pkl",
            #diffusion_steps = 50
        )
        else:
            results_eval=AF3Designer_model.predict(
            input_json_path=json_path,
            dump_dir=output_dir,
            seed=123,
            #input_atom_array_path="/storage/caolongxingLab/fangminchao/software/Protenix/coord_pdb.pkl",
            #diffusion_steps = 50
        )
        #print(output_dir)
        #print(results_eval)
        #import pickle

        #with open('/storage/caolab/fangmc/code/AF3Designer/results.pkl', 'wb') as f:
        #    pickle.dump(results_eval, f)
        max_ranking_score = 0
        for _ in range(len(results_eval['summary_confidence'])):
            ranking_score = results_eval['summary_confidence'][_]["ranking_score"]
            #print(ranking_score)
            if ranking_score > max_ranking_score:
                max_ranking = _
                max_ranking_score = ranking_score

        #print(max_ranking_result.metadata.keys())
        #print(max_ranking_result.numerical_data.keys())

        dict_result ={}
    
        dict_result['ipae'] = None  
        dict_result['ipde'] = None  

        dict_result['iptm'] = results_eval['summary_confidence'][max_ranking]['chain_iptm']
        dict_result['chain_pair_pae_min'] = None
        dict_result['chain_ptm'] = results_eval['summary_confidence'][max_ranking]['chain_ptm']
        dict_result['chain_pair_iptm'] = results_eval['summary_confidence'][max_ranking]['chain_pair_iptm']
        dict_result['ranking_confidence'] = results_eval['summary_confidence'][max_ranking]["ranking_score"]

        #print(dict_result)
        tag =f"{scaffold_basename}_{seq_count}".lower()
        cif_path = os.path.join(output_dir,tag,"seed_123","predictions",f"{tag}_seed_123_sample_0.cif")
        
        #confidence_json_path = os.path.join(output_dir,tag,f"{tag}_summary_confidences.json")
        #with open(confidence_json_path, 'r') as f:
        #    confidence_json = json.load(f)
        
        chain_labels = string.ascii_uppercase[:count]  # Generate labels A, B, C...
        
        metric["eval_status"] = "success"
        metric["eval_path"] = cif_path
        metric['prediction_model'] = "AF3"
        metric['eval_plddt'] = calculate_average_b_factor(cif_path,[f"{_}" for _ in chain_labels])
        if plddt_good_indicates:
            metric['eval_plddt_fix'] ,metric['eval_plddt_redes'] = calculate_bfactor_averages_from_list(cif_path,plddt_good_indicates)
        chain_pair_iptm = dict_result['chain_pair_iptm']
        chain_pair_pae = dict_result["chain_pair_pae_min"]
        metric['eval_iptm'] = dict_result['iptm'].mean().item()
        metric['eval_ptm'] = dict_result['chain_ptm'].mean().item()
        
        metric['eval_pae'] =  None
        metric['eval_pde'] =  None
        metric['eval_ipae'] = None
        metric['eval_ipde'] = None
        
        for i in range(0,count):
            label = chain_labels[i]
            metric[f'eval_{label}_plddt'] = calculate_average_b_factor(cif_path,[label])
            metric[f'eval_{label}_ptm'] = dict_result['chain_ptm'][i].item()
        print("rmsd")
        try:
            rmsd_result = calculate_ca_rmsd(cif_path, scaffold_path,fixed_chains)
            print(rmsd_result)
            _count =0
            _sm_count = 0
            #print(rmsd_result)
            for chain in chain_types:
                label = chain_labels[_count]
                if chain == "protein":
                    if fixed_chains:
                        metric[f'eval_protein_{label}_rmsd'] = rmsd_result['protein_rmsd'][_count]
                    else:
                        metric[f'eval_protein_{label}_rmsd'] = rmsd_result['protein_rmsd'][0]
                if chain == 'ligand':
                    metric[f'eval_ligand_{label}_rmsd'] = rmsd_result['ligand_rmsd'][_sm_count]
                    metric[f'eval_atom_distances_{label}'] = rmsd_result['atom_distances'][_sm_count]
                    _sm_count += 1
                if chain == 'dna':
                    metric[f'eval_dna_{label}_rmsd'] = rmsd_result['dna_rmsd']
                if chain == 'rna':
                    metric[f'eval_rna_{label}_rmsd'] = rmsd_result['rna_rmsd']
                _count += 1
            print("rmsd")
        except:
            print("rmsd wrong")
        if (sm_count != 0) or (rna_count != 0) or (dna_count!= 0):
            metric[f'eval_key_res_plddt'] = calculate_plddt_avg(cif_path, pocket_res)
            protein_indices = range(protein_count)
            other_indices = range(protein_count, count)  # Indices of other chains
            # Calculate iPTM of other chains relative to the protein chain
            all_iptm_to_protein = []
            all_ipae_to_protein = []
            for i in protein_indices:
                for j in other_indices:
                    all_iptm_to_protein.append(chain_pair_iptm[i][j])
                    all_iptm_to_protein.append(chain_pair_iptm[j][i])
            all_iptm_to_protein_val = [x for x in all_iptm_to_protein if x is not None]
            if all_iptm_to_protein:
                metric[f'eval_all_iptm_to_protein'] = (sum(all_iptm_to_protein_val) / len(all_iptm_to_protein_val)).item()
            else:
                metric[f'eval_all_iptm_to_protein'] = 0
            
            metric[f'eval_all_ipae_to_protein'] = 0
            for j in other_indices:
                label = chain_labels[j]
                cross_values_iptm = []
                for i in protein_indices:
                    cross_values_iptm.append(chain_pair_iptm[i][j])
                    cross_values_iptm.append(chain_pair_iptm[j][i])
                cross_values_iptm = [x for x in cross_values_iptm if x is not None]
                average_iptm = sum(cross_values_iptm) / len(cross_values_iptm) if cross_values_iptm else 0
                metric[f'eval_{label}_iptm'] = average_iptm.item()
                metric[f'eval_{label}_ipae'] = None

        seq_count += 1
        metrics_to_tile.append(metric)
    # ! remain to do, fix logits for reindex
    metrics_to_tile.sort(key=lambda x: x['eval_plddt'])
    return metrics_to_tile



def self_consistency_af3(scaffold_path,
                        AF3Designer_model,
                        output_dir,
                        template_path,
                        sm,  
                        ccd_LG,
                        ccd_codes,
                        dna,
                        rna,
                        ref_eval,
                        chain_types,
                        pocket_res,
                        fixed_chains,
                        plddt_good_indicates,
                        cyclic,
                        replace_MSA,
                        metrics):
    metrics_to_tile = []
    seq_count = 0
    for metric in metrics:
        cyclic_prediction = cyclic
        packed = metric["packed_path"]
        seq = metric['mpnn_sequence'] 
        if seq and packed and template_path:
            # Prepare input JSON for AF3
            with open(template_path, 'r') as f:
                template = json.load(f)
            input_json = template.copy()

            # Safely get the base name of the scaffold file, change extension to .fa
            scaffold_basename = os.path.splitext(os.path.basename(scaffold_path))[0]
            input_json['name'] = f"{scaffold_basename}_{seq_count}"
            json_filename = f"{scaffold_basename}_{seq_count}.json"
            json_path = os.path.join(output_dir, json_filename)
            input_json["modelSeeds"] = get_random_seeds(1)
            count = 0
            sm_count = 0
            rna_count = 0 
            dna_count = 0 
            protein_count = 0
            chain_labels = string.ascii_uppercase[:10] 
            for chain in chain_types:
                if chain == "protein":
                    if chain_labels[protein_count] not in fixed_chains:
                        input_json['sequences'][count]['protein']["sequence"] = seq.split(':')[protein_count]
                    protein_count +=1
                if chain == 'ligand':
                    if ccd_codes:
                        input_json['sequences'][count]["ligand"]["ccdCodes"] = [ccd_codes[sm_count]]
                    elif sm:
                        input_json['sequences'][count]["ligand"]["smiles"] = sm[sm_count]
                    sm_count +=1
                if chain == 'dna':
                    input_json['sequences'][count]["dna"]["sequence"] = dna[dna_count]
                    dna_count +=1
                if chain == 'rna':
                    input_json['sequences'][count]["rna"]["sequence"] = rna[rna_count]
                    rna_count +=1
                count += 1

            if ccd_LG:
                input_json['userCCD']=ccd_LG
        # Write the new JSON file
        if replace_MSA:
            #msa=f'>query\n{input_json["sequences"][chain]["protein"]["sequence"]}\n'
            #originMSA = input_json['sequences'][chain]['protein']["unpairedMsa"]
#
            #input_json['sequences'][chain]['protein']["unpairedMsa"] = msa
#
            ## Get old MSA
            origin_msa = input_json['sequences'][0]['protein']['unpairedMsa'].strip().splitlines()

            # new query sequence
            new_query_seq = input_json['sequences'][0]['protein']['sequence']

            # Find the first line starting with '>', which is the description line of the first sequence
            msa_lines = []
            i = 0
            while i < len(origin_msa):
                if origin_msa[i].startswith('>'):
                    if i == 0:  # Replace the first sequence
                        msa_lines.append('>query')
                        msa_lines.append(new_query_seq)
                        # Skip the sequence lines of the old query
                        i += 1
                        while i < len(origin_msa) and not origin_msa[i].startswith('>'):
                            i += 1
                        continue
                    else:
                        msa_lines.append(origin_msa[i])
                else:
                    msa_lines.append(origin_msa[i])
                i += 1

            # # Update input_json
            input_json['sequences'][0]['protein']['unpairedMsa'] = '\n'.join(msa_lines) + '\n'

        with open(json_path, 'w') as f:
            json.dump(input_json, f, indent=2)
            
        base_name = os.path.basename(scaffold_path)
       
        if ref_eval:
            print("normal AF3 batch diffusion plus ref guided diffusion")
            
            ref_out_dir = os.path.join(output_dir, "packed")
            result_pkl, file_name, error = process_single_file(( packed , ref_out_dir, ref_out_dir),None)
            # Run AF3
            pkl_path = os.path.join(ref_out_dir, f'{os.path.splitext(packed)[0]}.pkl')
            results_eval=AF3Designer_model.single_file_process(json_path,output_dir,
                            ref_pdb_path=pkl_path,
                            ref_time_steps =0,
                            num_samples=5,
                            ref_time_evaluation = 0,
                            cyclic =cyclic_prediction,
                            ref_pkl_dump_path=None)
        else:
            print("normal AF3 batch diffusion")
            results_eval=AF3Designer_model.single_file_process(json_path,output_dir,cyclic =cyclic_prediction)
        #print(results_eval)
        #import pickle

        # Save to file

        #with open('/storage/caolab/fangmc/code/AF3Designer/results.pkl', 'wb') as f:
        #    pickle.dump(results_eval, f)
        ranking_scores = []
        max_ranking_score = None
        max_ranking_result = None
        for results_for_seed in results_eval:
            seed = results_for_seed.seed
            for sample_idx, result in enumerate(results_for_seed.inference_results):
                #print(seed, sample_idx)
                ranking_score = float(result.metadata['ranking_score'])
                ranking_scores.append((seed, sample_idx, ranking_score))
                if max_ranking_score is None or ranking_score > max_ranking_score:
                    max_ranking_score = ranking_score
                    max_ranking_result = result


        #print(max_ranking_result.metadata.keys())

        dict_result ={}
        pae_all = max_ranking_result.numerical_data["full_pae"]
        pde_all = max_ranking_result.numerical_data["full_pde"]
        chains = np.array(max_ranking_result.metadata["token_chain_ids"])
        a_indices = np.where(chains == 'A')[0]  
        b_indices = np.where(chains == 'B')[0]  
        dict_result['pae'] = np.mean(pae_all)
        dict_result['pde'] = np.mean(pde_all)
        if len(b_indices) > 0:
            cross_pae_values = pae_all[np.ix_(a_indices, b_indices)] 
            cross_pde_values = pde_all[np.ix_(a_indices, b_indices)] 
            dict_result['ipae'] = np.mean(cross_pae_values)
            dict_result['ipde'] = np.mean(cross_pde_values)
        else:
            dict_result['ipae'] = None  # 
            dict_result['ipde'] = None  # 

        dict_result['iptm'] = max_ranking_result.metadata["iptm"]
        dict_result['chain_pair_pae_min'] = max_ranking_result.metadata["chain_pair_pae_min"]
        dict_result['ptm'] = max_ranking_result.metadata["ptm"]
        dict_result['chain_pair_iptm'] = max_ranking_result.metadata["chain_pair_iptm"]
        dict_result['ranking_confidence'] = max_ranking_result.metadata["ranking_confidence"]

        dict_result['chain_ptm'] = max_ranking_result.metadata["iptm_ichain"]
        #print(dict_result)
        tag =f"{scaffold_basename}_{seq_count}".lower()
        cif_path = os.path.join(output_dir,tag,f"{tag}_model.cif")
        
        chain_labels = string.ascii_uppercase[:count]  # Generate labels A, B, C...
        
        metric["eval_status"] = "success"
        metric["eval_path"] = cif_path
        metric['prediction_model'] = "AF3"
        metric['eval_plddt'] = calculate_average_b_factor(cif_path,[f"{_}" for _ in chain_labels])
        if plddt_good_indicates:
            metric['eval_plddt_fix'] ,metric['eval_plddt_redes'] = calculate_bfactor_averages_from_list(cif_path,plddt_good_indicates)
        chain_pair_iptm = dict_result['chain_pair_iptm']
        chain_pair_pae = dict_result["chain_pair_pae_min"]
        metric['eval_iptm'] = dict_result['iptm']
        metric['eval_ptm'] = dict_result['ptm']
        
        metric['eval_pae'] = dict_result['pae'] 
        metric['eval_pde'] = dict_result['pde'] 
        metric['eval_ipae'] = dict_result['ipae'] 
        metric['eval_ipde'] = dict_result['ipde'] 
        
        for i in range(0,count):
            label = chain_labels[i]
            metric[f'eval_{label}_plddt'] = calculate_average_b_factor(cif_path,[label])
            metric[f'eval_{label}_ptm'] = dict_result['chain_ptm'][i]
        else:
            try:
                rmsd_result = calculate_ca_rmsd(cif_path, scaffold_path,fixed_chains)
                _count =0
                _sm_count = 0
                #print(rmsd_result)
                for chain in chain_types:
                    label = chain_labels[_count]
                    if chain == "protein":
                        if fixed_chains:
                            metric[f'eval_protein_{label}_rmsd'] = rmsd_result['protein_rmsd'][_count]
                        else:
                            metric[f'eval_protein_{label}_rmsd'] = rmsd_result['protein_rmsd'][0]
                    if chain == 'ligand':
                        metric[f'eval_ligand_{label}_rmsd'] = rmsd_result['ligand_rmsd'][_sm_count]
                        metric[f'eval_atom_distances_{label}'] = rmsd_result['atom_distances'][_sm_count]
                        _sm_count += 1
                    if chain == 'dna':
                        metric[f'eval_dna_{label}_rmsd'] = rmsd_result['dna_rmsd']
                    if chain == 'rna':
                        metric[f'eval_rna_{label}_rmsd'] = rmsd_result['rna_rmsd']
                    _count += 1
            except:
                print("rmsd wrong")

            if (sm_count != 0) or (rna_count != 0) or (dna_count!= 0):
                metric[f'eval_key_res_plddt'] = calculate_plddt_avg(cif_path, pocket_res)
                protein_indices = range(protein_count)
                other_indices = range(protein_count, count)  # Indices of other chains
                # Calculate iPTM of other chains relative to the protein chain
                all_iptm_to_protein = []
                all_ipae_to_protein = []
                for i in protein_indices:
                    for j in other_indices:
                        all_iptm_to_protein.append(chain_pair_iptm[i][j])
                        all_ipae_to_protein.append(chain_pair_pae[i][j])
                        all_iptm_to_protein.append(chain_pair_iptm[j][i])
                        all_ipae_to_protein.append(chain_pair_pae[j][i])
                all_iptm_to_protein_val = [x for x in all_iptm_to_protein if x is not None]
                all_ipae_to_protein_val = [x for x in all_ipae_to_protein if x is not None]
                if all_iptm_to_protein:
                    metric[f'eval_all_iptm_to_protein'] = sum(all_iptm_to_protein_val) / len(all_iptm_to_protein_val)
                else:
                    metric[f'eval_all_iptm_to_protein'] = 0
                if all_ipae_to_protein:
                    metric[f'eval_all_ipae_to_protein'] = sum(all_ipae_to_protein_val) / len(all_ipae_to_protein_val)
                else:
                    metric[f'eval_all_ipae_to_protein'] = 0
                for j in other_indices:
                    label = chain_labels[j]
                    cross_values_iptm = []
                    cross_values_ipae = []
                    for i in protein_indices:
                        cross_values_iptm.append(chain_pair_iptm[i][j])
                        cross_values_ipae.append(chain_pair_pae[i][j])
                        cross_values_iptm.append(chain_pair_iptm[j][i])
                        cross_values_ipae.append(chain_pair_pae[j][i])
                    cross_values_ipae = [x for x in cross_values_ipae if x is not None]
                    cross_values_iptm = [x for x in cross_values_iptm if x is not None]
                    average_iptm = sum(cross_values_iptm) / len(cross_values_iptm) if cross_values_iptm else 0
                    average_ipae = sum(cross_values_ipae) / len(cross_values_ipae) if cross_values_ipae else 0
                    metric[f'eval_{label}_iptm'] = average_iptm
                    metric[f'eval_{label}_ipae'] = average_ipae

        seq_count += 1
        metrics_to_tile.append(metric)
    # ! remain to do, fix logits for reindex
    metrics_to_tile.sort(key=lambda x: x['eval_plddt'])
    return metrics_to_tile

import subprocess
def protenix_eval(input_json_path,dump_dir,seed,diffusion_steps,use_esm):
    command_string = (
        f"~/mambaforge/envs/protenix/bin/python /storage/caolongxingLab/fangminchao/AF3_design/AF3Designer/protenix_init.py "
        f"--input_json_path {input_json_path} "
        f"--dump_dir {dump_dir} "
        f"--seed {seed} " # seed and diffusion_steps are ints, no need to quote if they are simple numbers
        f"--diffusion_steps {diffusion_steps}"
    )
    if use_esm:
        command_string += " --use_esm"

    print(f"Running command: {command_string}")

    try:
        result = subprocess.run(
            command_string,
            shell=True, # VERY IMPORTANT for running as a single string
            capture_output=True,
            text=True,
            check=True
        )
        print("STDOUT:\n", result.stdout)
        if result.stderr:
            print("STDERR:\n", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
    except FileNotFoundError:
        print(f"Error: Shell or command not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")




def run_mpnn_evaluation(scaffold_path,
                        mpnn_model,
                        mpnn_config_dict,
                        plddt_good_indicates,
                        pocket_res,
                        output_dir,
                        symmetry_residues,
                        metrics,
                        cyclic,
                        cycle,
                        evaluator= None,
                        bais_per_residues=None):  
    #pocket_plddt_good_res = common_elements(pocket_res ,plddt_good_indicates)
    #pocket_plddt_good_res_set = set(pocket_plddt_good_res)
    pocket_res_to_fix = [item for item in plddt_good_indicates]
    non_pocket_to_fix =  list(set(plddt_good_indicates + pocket_res)) 
    pocket_res_to_fix = " ".join([f"{resi}" for  resi in pocket_res_to_fix]) 
    non_pocket_to_fix = " ".join([f"{resi}" for  resi in non_pocket_to_fix]) 
    print("plddt_good_indicates",plddt_good_indicates)
    print("pocket_res",pocket_res)
    print("pocket_res_to_fix",pocket_res_to_fix)
    print("non_pocket_to_fix",non_pocket_to_fix)
    sequences = []
    packed_paths =[]
    residue_groups = parse_symmetry_residues(symmetry_residues)
    weights = calculate_weights(residue_groups)
    weights_str = '|'.join([','.join([str(weight)] * len(group)) for weight, group in zip(weights, residue_groups)])
    bias_AA = ""
    if cyclic:
        bias_AA = "D:0.5,E:0.5,H:0.5,K:0.5,R:0.5,W:-0.5,L:-0.5,I:-0.5,F:-0.5,M:-0.5,V:-0.5,Y:-0.5"

    if evaluator :
        mpnn_config_dict["num_seqs"] = int(mpnn_config_dict["num_seqs"] *8)

    if mpnn_config_dict["model_name"] =="ligandmpnn_plus_proteinmpnn":
        print("ligandmpnn plus proteinmpnn evaluation")
        sequences,packed_paths=run_Ligandmpnn_plus_proteinmpnn_evaluation(mpnn_model,scaffold_path,mpnn_config_dict,pocket_res_to_fix,non_pocket_to_fix,interact_fix_analyzer,weights_str,output_dir,symmetry_residues)
    else: 
        sequences,packed_paths=run_purempnn_evaluation(mpnn_model,scaffold_path,mpnn_config_dict,pocket_res_to_fix,weights_str,output_dir,bias_AA,symmetry_residues,bais_per_residues)

    if isinstance(evaluator, CoDP):
        mpnn_config_dict["num_seqs"] = int(mpnn_config_dict["num_seqs"] /8)
        num_seqs = int(mpnn_config_dict["num_seqs"])
        #scores = evaluator.predict(sequences, scaffold_path)
        #interaction_data = [(seq, packed, 0, 0, 0, score) for seq, packed, score in zip(sequences, packed_paths, scores)]
        #original_length_sequences = sorted(interaction_data, key=lambda x: -x[5])[:num_seqs]
        batchsizes = [ 8, 4, 2]
        last_error = None

        for batchsize in batchsizes:
            try:
                original_length_sequences = run_selection_process(
                    sequences, packed_paths, evaluator, scaffold_path, num_seqs, batchsize,
                )
                print(f"✅ Successfully ran with batchsize: {batchsize}")
                break  # Break loop on success
            except Exception as e:
                last_error = e
                print(f"⚠️ Batchsize {batchsize} failed: {e}. Trying smaller batchsize...")
        else:  # Loop finished normally (all batches failed)
            print("❌ All batchsizes failed. Raising last error.")
            raise last_error
    else:
        original_length_sequences = [(seq, packed, 0, 0, 0, 0) for seq, packed in zip(sequences, packed_paths)]

    metrics_to_tile = []
    metrics_to_copy = copy.deepcopy(metrics)

    for seq, packed, plip_score, pi_stacking_score, oxygen_score, score in original_length_sequences:
        metrics = copy.deepcopy(metrics_to_copy)
        metrics["mpnn_model"] = mpnn_config_dict["model_name"]
        metrics["packed_path"] = packed
        metrics['mpnn_sequence'] = seq
        metrics["eval_status"] = "Not run"
        metrics['esm_score'] = score
        metrics_to_tile.append(metrics)
    return metrics_to_tile


import torch
def print_cuda_memory_usage(stage=""):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)

    print(f"\n--- CUDA Memory Usage [{stage}] ---")
    print(f"Device: {device_name}")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Reserved:    {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Max Reserved:    {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB")
    print("-" * 30)

import time
def run_selection_process(sequences, packed_paths, evaluator, scaffold_path, num_seqs,batch_size):
    """
    Performs multi-round, knockout-style filtering based on scores.

    Args:
        sequences (list): List of generated amino acid sequences.
        packed_paths (list): List of PDB file paths corresponding to the sequences.
        evaluator (object): The evaluator object used to predict scores.
        scaffold_path (str): Path to the scaffold PDB file.
        mpnn_config_dict (dict): Dictionary containing the num_seqs configuration.

    Returns:
        list: Filtered list of (seq, packed, score) tuples, sorted by score in descending order.
    """
    current_data = [(seq, packed) for seq, packed in zip(sequences, packed_paths)]
    #print(current_data)
    # Target sequence count
    target_num_seqs = num_seqs # Get the final desired sequence count
    start_time = time.time()
    print(f"Starting multi-round sequence filtering, initial sequence count: {len(current_data)}")
    print(f"Target final sequence count: {target_num_seqs}")
    # Ensure target_num_seqs is at least 1
    if target_num_seqs == 0:
        target_num_seqs = 1 
    while len(current_data) > target_num_seqs:
        batch_size = 4 # Maximum number of sequences to process per batch
        all_scores_for_round = []
        
        # Perform prediction in batches
        for i in range(0, len(current_data), batch_size):
            batch_sequences = [item[0] for item in current_data[i : i + batch_size]]
            batch_packed_paths = [item[1] for item in current_data[i : i + batch_size]]
            # Call the evaluator for prediction
            # Note: evaluator.predict should return scores corresponding to the order of batch_sequences
            #print_cuda_memory_usage()
            #torch.cuda.empty_cache()
            #print_cuda_memory_usage()
            batch_scores = evaluator.predict(batch_sequences, scaffold_path)
            #print_cuda_memory_usage()
            #torch.cuda.empty_cache()
            # Bind scores with original sequences and paths
            for j, (seq, packed) in enumerate(zip(batch_sequences, batch_packed_paths)):
                all_scores_for_round.append((seq, packed, batch_scores[j]))
        
        # Sort all sequences in this round by score in descending order
        all_scores_for_round.sort(key=lambda x: -x[2]) # x[2] is score
        # Filter out the top half of sequences to enter the next round
        # If the count is odd, take (N+1)/2 to ensure at least half
        next_round_count = max(target_num_seqs, (len(all_scores_for_round) + 1) // 2)
        
        # If the number of sequences for the next round is greater than the current total, use the current total
        # If the current total is less than the target number, use the target number, but this is already handled by the loop condition
        
        current_data = [(item[0], item[1]) for item in all_scores_for_round[:next_round_count]]
        
        print(f"Number of sequences remaining after this round of filtering: {len(current_data)}")
        if len(current_data) == 0: # Avoid infinite loop or empty list
            print("Warning: Sequence list is empty after filtering, terminating early.")
            break
    
    # The final result needs to contain the complete interaction_data format (seq, packed, 0, 0, 0, score)
    # Here, 'score' should be the final score, and the other fields are filled with 0
    final_results = []
    for seq, packed in current_data:
        # Find the final score corresponding to the sequence (simplified to 0 here; requires storage for actual scores)
        # Theoretically, 'all_scores_for_round' stores the last round's scores, which could be used.
        # To ensure the format matches the original 'original_length_sequences', we temporarily fill score=0 here.
        # If the result from 'evaluator.predict' is the final score, it can be filled in here.

        # Look up the final score for the corresponding sequence (this would be slightly complicated, requiring traceback)
        # For simplicity, we assume only the final sequence and path are needed here, and score can be filled with 0 or the last round's score.
        # Assume 'all_scores_for_round' from the final round contains the definitive results.

        # Look up the score of the current sequence in the final round
        final_score = 0
        for item in all_scores_for_round: # Ensure all_scores_for_round is the complete list from the final round
            if item[0] == seq and item[1] == packed:
                final_score = item[2]
                break
        
        final_results.append((seq, packed, 0, 0, 0, final_score))
    # The final results still need to be sorted by score in descending order (even if they were already sorted in the loop)
    final_results.sort(key=lambda x: -x[5])
    
    print(f"Final filtering complete, remaining sequence count: {len(final_results)}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Code execution time: {elapsed_time} seconds")
    return final_results



def run_Ligandmpnn_plus_proteinmpnn_evaluation(mpnn_model,scaffold_path,mpnn_config_dict,pocket_res_to_fix,non_pocket_to_fix,interact_fix_analyzer,weights_str,output_dir,symmetry_residues):  
    # ! to do 
    protein_mpnn , ligand_mpnn =  mpnn_model
    sequences  = []
    packed_paths = []
    print(f"{mpnn_config_dict['model_name']} design")
    output_dir_mpnn =os.path.join(output_dir,f"{mpnn_config_dict['model_name']}")
    sequences_stack,pdb_path_stack=ligand_mpnn.single_protein_mpnn_design(scaffold_path=scaffold_path, 
                                      output_dir_mpnn=output_dir_mpnn, 
                                      numbers_seqs=mpnn_config_dict['num_seqs'],   
                                      chains_to_design="",
                                      fixed_res=pocket_res_to_fix, 
                                      redesigned_residues="", 
                                      symmetry_residues=symmetry_residues,
                                      weights_str=weights_str,)
    sequences,packed_paths = [],[]
    for pdb in pdb_path_stack:
        sequences_stack_prompnn,pdb_path_stack_prompnn=protein_mpnn.single_protein_mpnn_design(scaffold_path=pdb, 
                                      output_dir_mpnn=output_dir_mpnn, 
                                      numbers_seqs=1,   
                                      chains_to_design="",
                                      fixed_res=non_pocket_to_fix, 
                                      redesigned_residues="", 
                                      symmetry_residues=symmetry_residues,
                                      weights_str=weights_str,)
        sequences.append(sequences_stack_prompnn[0])
        packed_paths.append(pdb_path_stack_prompnn[0])

    return  sequences,packed_paths

def run_purempnn_evaluation(mpnn_model,scaffold_path,mpnn_config_dict,pocket_res_to_fix,weights_str,output_dir,bias_AA,symmetry_residues,bais_per_residues):  
    

    print(f"{mpnn_config_dict['model_name']} design")
    output_dir_mpnn =os.path.join(output_dir,f"{mpnn_config_dict['model_name']}")
    sequences_stack,pdb_path_stack=mpnn_model.single_protein_mpnn_design(scaffold_path=scaffold_path, 
                                      output_dir_mpnn=output_dir_mpnn, 
                                      numbers_seqs=mpnn_config_dict['num_seqs'],   
                                      chains_to_design="",
                                      fixed_res=pocket_res_to_fix, 
                                      redesigned_residues="", 
                                      symmetry_residues=symmetry_residues,
                                      bais_per_residues = bais_per_residues,
                                      input_bias_AA = bias_AA,
                                      weights_str=weights_str,)

    return  sequences_stack,pdb_path_stack


def get_fake_msa(file_path: str,fake_msa,chain,mpnn_model):
    chain_labels = string.ascii_uppercase[:10] 
    id = chain_labels[chain]
    if isinstance(mpnn_model, list):
        mpnn_model = mpnn_model[0]
        
    sequences_stack,pdb_path_stack=mpnn_model.single_protein_mpnn_design(scaffold_path=file_path, 
                                      output_dir_mpnn="", 
                                      numbers_seqs=fake_msa,   chains_to_design=id,
                                      pack_side_chains_or_not=True)
    
    sequences=sequences_stack
    #print(sequences)
    fake_msa = ""
    msa = []
    for i, seq in enumerate(sequences, 1):
        msa.append(f">Seq{i}")
        msa.append(seq.split(":")[chain])
    return "\n".join(msa)+"\n"
        

def process_confidence_metrics_protenix(results_op, cif_path: str, copied_file : str,scaffold_path, metrics_tile,pocket_res,chain_types,fixed_chains,count_tuple) -> Dict:
    """Process confidence metrics output by AF3"""
    try:
        #print(results_op)
        max_ranking_score = -1000
        for _ in range(len(results_op['summary_confidence'])):
            ranking_score = results_op['summary_confidence'][_]["ranking_score"]
            #print(ranking_score)
            if ranking_score > max_ranking_score:
                max_ranking = _
                max_ranking_score = ranking_score
        #print(max_ranking)
        #print(max_ranking_result.metadata.keys())
        #print(max_ranking_result.numerical_data.keys())

        dict_result ={}
    
        dict_result['ipae'] = None 
        dict_result['ipde'] = None 

        dict_result['iptm'] = results_op['summary_confidence'][max_ranking]['chain_iptm']
        dict_result['chain_pair_pae_min'] = None
        dict_result['chain_ptm'] = results_op['summary_confidence'][max_ranking]['chain_ptm']
        dict_result['chain_pair_iptm'] = results_op['summary_confidence'][max_ranking]['chain_pair_iptm']
        dict_result['ranking_confidence'] = results_op['summary_confidence'][max_ranking]["ranking_score"]

        #print(dict_result)
        
        #confidence_json_path = os.path.join(output_dir,tag,f"{tag}_summary_confidences.json")
        #with open(confidence_json_path, 'r') as f:
        #    confidence_json = json.load(f)
        metrics ={}
        metrics['op_cif_path'] = cif_path
        [count,protein_count,sm_count,rna_count,dna_count] =count_tuple 
        chain_labels = string.ascii_uppercase[:count]  # Generate labels A, B, C...
        metrics["AF3_Status"] = "success"
        metrics['op_plddt'] = calculate_average_b_factor(cif_path,[f"{_}" for _ in chain_labels])
        chain_pair_iptm = dict_result['chain_pair_iptm']
        chain_pair_pae = dict_result['chain_pair_pae_min']
        metrics['op_iptm'] = dict_result['iptm'].mean().item()
        metrics['op_ptm'] = dict_result['chain_ptm'].mean().item()
        metrics['op_pae'] = None
        metrics['op_pde'] = None
        metrics['op_ipae'] = dict_result['ipae'] 
        metrics['op_ipde'] = dict_result['ipde'] 
        metrics["AF3_Status"] = "success"
        metrics['op_plddt'] = calculate_average_b_factor(cif_path,[f"{_}" for _ in chain_labels])
        
        for i in range(0,count):
            label = chain_labels[i]
            metrics[f'op_{label}_plddt'] = calculate_average_b_factor(cif_path,[label])
            metrics[f'op_{label}_ptm'] = dict_result['chain_ptm'][i].item()
        
        try:
            rmsd_result = calculate_ca_rmsd(cif_path, copied_file,fixed_chains)
            rmsd_result_to_origin = calculate_ca_rmsd(cif_path, scaffold_path,fixed_chains)
            _count =0
            _sm_count = 0

            for chain in chain_types:
                label = chain_labels[_count]
                if chain == "protein":
                    if fixed_chains:
                        metrics[f'op_protein_{label}_rmsd'] = rmsd_result['protein_rmsd'][_count]
                        metrics[f'origin_protein_{label}_rmsd'] = rmsd_result_to_origin['protein_rmsd'][_count]
                    else:
                        metrics[f'op_protein_{label}_rmsd'] = rmsd_result['protein_rmsd'][0]
                        metrics[f'origin_protein_{label}_rmsd'] = rmsd_result_to_origin['protein_rmsd'][0]
                if chain == 'ligand':
                    metrics[f'op_ligand_{label}_rmsd'] = rmsd_result['ligand_rmsd'][_sm_count]
                    metrics[f'op_atom_distances_{label}'] = rmsd_result['atom_distances'][_sm_count]
                    metrics[f'origin_ligand_{label}_rmsd'] = rmsd_result_to_origin['ligand_rmsd'][_sm_count]
                    metrics[f'origin_atom_distances_{label}'] = rmsd_result_to_origin['atom_distances'][_sm_count]
                    _sm_count += 1
                if chain == 'dna':
                    metrics[f'op_dna_{label}_rmsd'] = rmsd_result['dna_rmsd']
                    metrics[f'origin_dna_{label}_rmsd'] = rmsd_result_to_origin['dna_rmsd']
                if chain == 'rna':
                    metrics[f'op_rna_{label}_rmsd'] = rmsd_result['rna_rmsd']
                    metrics[f'origin_rna_{label}_rmsd'] = rmsd_result_to_origin['rna_rmsd']
                _count += 1 
        except:
            print("rmsd wrong")
        
        if (sm_count != 0) or (rna_count != 0) or (dna_count!= 0):
            metrics[f'eval_key_res_plddt'] = calculate_plddt_avg(cif_path, pocket_res)
            protein_indices = range(protein_count)
            other_indices = range(protein_count, count)  # Indices of other chains
            # Calculate iPTM of other chains relative to the protein chain
            all_iptm_to_protein = []
            all_ipae_to_protein = []
            for i in protein_indices:
                for j in other_indices:
                    all_iptm_to_protein.append(chain_pair_iptm[i][j])

                    all_iptm_to_protein.append(chain_pair_iptm[j][i])

            all_iptm_to_protein = [x for x in all_iptm_to_protein if x is not None]

            if all_iptm_to_protein:
                metrics[f'eval_all_iptm_to_protein'] = (sum(all_iptm_to_protein) / len(all_iptm_to_protein)).item()
            else:
                metrics[f'eval_all_iptm_to_protein'] = 0
                
            metrics[f'eval_all_ipae_to_protein'] = 0
            
            for j in other_indices:
                label = chain_labels[j]
                cross_values_iptm = []
                cross_values_ipae = []
                for i in protein_indices:
                    cross_values_iptm.append(chain_pair_iptm[i][j])
                    cross_values_iptm.append(chain_pair_iptm[j][i])

                cross_values_iptm = [x for x in cross_values_iptm if x is not None]
                average_iptm = sum(cross_values_iptm) / len(cross_values_iptm) if cross_values_iptm else 0
                metrics[f'eval_{label}_iptm'] = average_iptm.item()
                metrics[f'eval_{label}_ipae'] = None

        for _ in metrics_tile:
            _.update(metrics)
        return metrics_tile
    
    except Exception as e:
        print(f"Failed to process AF3 metrics: {str(e)}")
        return metrics

def process_confidence_metrics(results_op, cif_path: str, copied_file : str,scaffold_path, metrics_tile,pocket_res,chain_types,fixed_chains,count_tuple) -> Dict:
    """Process confidence metrics output by AF3"""
    try:
        
        ranking_scores = []
        max_ranking_score = None
        max_ranking_result = None
        for results_for_seed in results_op:
            seed = results_for_seed.seed
            for sample_idx, result in enumerate(results_for_seed.inference_results):
                #print(seed, sample_idx)
                ranking_score = float(result.metadata['ranking_score'])
                ranking_scores.append((seed, sample_idx, ranking_score))
                if max_ranking_score is None or ranking_score > max_ranking_score:
                    max_ranking_score = ranking_score
                    max_ranking_result = result


        #print(max_ranking_result.metadata.keys())
        #print(max_ranking_result.numerical_data.keys())

        dict_result ={}
        pae_all = max_ranking_result.numerical_data["full_pae"]
        pde_all = max_ranking_result.numerical_data["full_pde"]
        chains = np.array(max_ranking_result.metadata["token_chain_ids"])
        a_indices = np.where(chains == 'A')[0]  
        b_indices = np.where(chains == 'B')[0]  

        dict_result['pae'] = np.mean(pae_all)
        dict_result['pde'] = np.mean(pde_all)

        if len(b_indices) > 0:
            cross_pae_values = pae_all[np.ix_(a_indices, b_indices)] 
            cross_pde_values = pde_all[np.ix_(a_indices, b_indices)] 
            dict_result['ipae'] = np.mean(cross_pae_values)
            dict_result['ipde'] = np.mean(cross_pde_values)
        else:
            dict_result['ipae'] = None  
            dict_result['ipde'] = None  
        
        dict_result['iptm'] = max_ranking_result.metadata["iptm"]
        dict_result['chain_pair_pae_min'] = max_ranking_result.metadata["chain_pair_pae_min"]
        dict_result['ptm'] = max_ranking_result.metadata["ptm"]
        dict_result['chain_pair_iptm'] = max_ranking_result.metadata["chain_pair_iptm"]
        dict_result['ranking_confidence'] = max_ranking_result.metadata["ranking_confidence"]

        dict_result['chain_ptm'] = max_ranking_result.metadata["iptm_ichain"]
        metrics ={}
        metrics['op_cif_path'] = cif_path
        [count,protein_count,sm_count,rna_count,dna_count] =count_tuple 
        chain_labels = string.ascii_uppercase[:count]  # Generate labels A, B, C...
        metrics["AF3_Status"] = "success"
        metrics['op_plddt'] = calculate_average_b_factor(cif_path,[f"{_}" for _ in chain_labels])
        chain_pair_iptm = dict_result['chain_pair_iptm']
        chain_pair_pae = dict_result['chain_pair_pae_min']
        metrics['op_iptm'] = dict_result['iptm']
        metrics['op_ptm'] = dict_result['ptm'] 
        metrics['op_pae'] = dict_result['pae'] 
        metrics['op_pde'] = dict_result['pde'] 
        metrics['op_ipae'] = dict_result['ipae'] 
        metrics['op_ipde'] = dict_result['ipde'] 
        for i in range(0,count):
            label = chain_labels[i]
            metrics[f'op_{label}_plddt'] = calculate_average_b_factor(cif_path,[label])
            metrics[f'op_{label}_ptm'] = dict_result['chain_ptm'][i]
        try:
            rmsd_result = calculate_ca_rmsd(cif_path, copied_file,fixed_chains)
            rmsd_result_to_origin = calculate_ca_rmsd(cif_path, scaffold_path,fixed_chains)
            _count =0
            _sm_count = 0

            for chain in chain_types:
                label = chain_labels[_count]
                if chain == "protein":
                    if fixed_chains:
                        metrics[f'op_protein_{label}_rmsd'] = rmsd_result['protein_rmsd'][_count]
                        metrics[f'origin_protein_{label}_rmsd'] = rmsd_result_to_origin['protein_rmsd'][_count]
                    else:
                        metrics[f'op_protein_{label}_rmsd'] = rmsd_result['protein_rmsd'][0]
                        metrics[f'origin_protein_{label}_rmsd'] = rmsd_result_to_origin['protein_rmsd'][0]
                if chain == 'ligand':
                    metrics[f'op_ligand_{label}_rmsd'] = rmsd_result['ligand_rmsd'][_sm_count]
                    metrics[f'op_atom_distances_{label}'] = rmsd_result['atom_distances'][_sm_count]
                    metrics[f'origin_ligand_{label}_rmsd'] = rmsd_result_to_origin['ligand_rmsd'][_sm_count]
                    metrics[f'origin_atom_distances_{label}'] = rmsd_result_to_origin['atom_distances'][_sm_count]
                    _sm_count += 1
                if chain == 'dna':
                    metrics[f'op_dna_{label}_rmsd'] = rmsd_result['dna_rmsd']
                    metrics[f'origin_dna_{label}_rmsd'] = rmsd_result_to_origin['dna_rmsd']
                if chain == 'rna':
                    metrics[f'op_rna_{label}_rmsd'] = rmsd_result['rna_rmsd']
                    metrics[f'origin_rna_{label}_rmsd'] = rmsd_result_to_origin['rna_rmsd']
                _count += 1 
        except:
            print("rmsd wrong")
        
        if (sm_count != 0) or (rna_count != 0) or (dna_count!= 0):
            metrics[f'eval_key_res_plddt'] = calculate_plddt_avg(cif_path, pocket_res)
            protein_indices = range(protein_count)
            other_indices = range(protein_count, count)  # Indices of other chains
            # Calculate iPTM of other chains relative to the protein chain
            all_iptm_to_protein = []
            all_ipae_to_protein = []
            for i in protein_indices:
                for j in other_indices:
                    all_iptm_to_protein.append(chain_pair_iptm[i][j])
                    all_ipae_to_protein.append(chain_pair_pae[i][j])
                    all_iptm_to_protein.append(chain_pair_iptm[j][i])
                    all_ipae_to_protein.append(chain_pair_pae[j][i])
            all_iptm_to_protein = [x for x in all_iptm_to_protein if x is not None]
            all_ipae_to_protein = [x for x in all_ipae_to_protein if x is not None]
            if all_iptm_to_protein:
                metrics[f'eval_all_iptm_to_protein'] = sum(all_iptm_to_protein) / len(all_iptm_to_protein)
            else:
                metrics[f'eval_all_iptm_to_protein'] = 0
                
            if all_ipae_to_protein:
                metrics[f'eval_all_ipae_to_protein'] = sum(all_ipae_to_protein) / len(all_ipae_to_protein)
            else:
                metrics[f'eval_all_ipae_to_protein'] = 0
            
            for j in other_indices:
                label = chain_labels[j]
                cross_values_iptm = []
                cross_values_ipae = []
                for i in protein_indices:
                    cross_values_iptm.append(chain_pair_iptm[i][j])
                    cross_values_ipae.append(chain_pair_pae[i][j])
                    cross_values_iptm.append(chain_pair_iptm[j][i])
                    cross_values_ipae.append(chain_pair_pae[j][i])
                cross_values_ipae = [x for x in cross_values_ipae if x is not None]
                cross_values_iptm = [x for x in cross_values_iptm if x is not None]
                average_iptm = sum(cross_values_iptm) / len(cross_values_iptm) if cross_values_iptm else 0
                average_ipae = sum(cross_values_ipae) / len(cross_values_ipae) if cross_values_ipae else 0
                metrics[f'eval_{label}_iptm'] = average_iptm
                metrics[f'eval_{label}_ipae'] = average_ipae

        for _ in metrics_tile:
            _.update(metrics)
        return metrics_tile
    
    except Exception as e:
        print(f"Failed to process AF3 metrics: {str(e)}")
        return metrics

class CoDP():
    def __init__(self,checkpoints_to_run,esm_name):
        bins_setting = {
        'first_break': 2.3125,
        'last_break': 21.6875,
        'num_bins': 8
        }
        crop_size = 256
        print("Model loading...")

        # Assuming contactModel is your defined model class
        self.contact_model = ContactModel(
            esm_name, 
            input_channels=384, 
            n_filters=256, 
            kernel_size=3, 
            n_layers=8,
            num_bins=bins_setting['num_bins'],
            crop_size=crop_size
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.contact_model.to(device)
        checkpoint = torch.load(checkpoints_to_run, map_location=device)
        saved_state_dict = {
            k: v for k, v in checkpoint.items()
            if k in [name for name, param in self.contact_model.named_parameters() if param.requires_grad]
        }

        self.contact_model.load_state_dict(saved_state_dict, strict=False)
    
    def predict(self,sequneces,pdb_path):
        #backbone_array = []
        #for _ in pdb_path:
        #    backbone = extract_pdb_info(_)
        #    backbone_array.append(backbone)
        print(pdb_path)
        backbone = extract_pdb_info(pdb_path)
        print(backbone.shape)
        backbone_with_batch = np.expand_dims(backbone, axis=0)  # Add a dimension at axis 0, transforming the shape to (1, L, 3, 3)
        backbone_with_batch = np.repeat(backbone_with_batch, len(sequneces), axis=0)  # Repeat this dimension 'batch_size' times, transforming it to (batch_size, L, 3, 3)
        backbone_with_batch = torch.tensor(backbone_with_batch, dtype=torch.float32)
        
        backbone =  compute_rbf(backbone_with_batch)
        scores = self.contact_model(sequneces,crop_size=600, true_contact=backbone, validation = True)
        scores = scores.mean(dim=1).tolist()
        return scores

from transformers.models.esm.modeling_esm import EsmForMaskedLM
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

def _rbf(D):
    device = D.device
    D_min, D_max, D_count = 2.0, 22.0, 16
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, 1, 1, 1, -1])  # Adjust shape for broadcasting
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)  # Expand last dimension for RBF
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF

def compute_rbf(backbone):
    """
    Generate B L L D tensor from backbone data.
    
    Parameters:
    backbone (torch.Tensor): A tensor of shape [B, L, 3, 3] representing the backbone coordinates.
    
    Returns:
    torch.Tensor: A tensor of shape [B, L, L, 3 * num_rbf].
    """
    # Backbone shape: [B, L, 3, 3]
    B, L, _, _ = backbone.shape
    # Step 1: Compute pairwise distances for each atom (N, CA, C)
    D_N = torch.sqrt(torch.sum((backbone[:, :, None, 0, :] - backbone[:, None, :, 0, :]) ** 2, -1) + 1e-6)
    D_CA = torch.sqrt(torch.sum((backbone[:, :, None, 1, :] - backbone[:, None, :, 1, :]) ** 2, -1) + 1e-6)
    D_C = torch.sqrt(torch.sum((backbone[:, :, None, 2, :] - backbone[:, None, :, 2, :]) ** 2, -1) + 1e-6)
    # Step 2: Stack distances along a new dimension
    D_combined = torch.stack([D_N, D_CA, D_C], dim=-1)  # Shape: [B, L, L, 3]
    # Step 3: Apply radial basis function (RBF) transformation
    RBF = _rbf(D_combined)  # Shape: [B, L, L, 3, num_rbf]
    # Step 4: Flatten the last two dimensions
    RBF = RBF.view(B, L, L, -1)  # Shape: [B, L, L, 3 * num_rbf]
    return RBF

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate=1, dropout_rate=0.15):
        super().__init__()
        # Calculate padding value
        padding = (dilation_rate * (kernel_size - 1)) // 2

        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=padding, 
            dilation=dilation_rate
        )
        self.inst_norm1 = nn.InstanceNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation_rate
        )
        self.inst_norm2 = nn.InstanceNorm2d(num_features=out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Add a residual connection if input and output channels differ
        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        residual = x
        if self.residual_conv is not None:
            residual = self.residual_conv(x)
            
        out = self.conv1(x)
        out = self.inst_norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.inst_norm2(out)
        out += residual
        out = F.relu(out)
        return out

class ConvPoolToFixedDim(nn.Module):
    def __init__(self, n_filters):
        super(ConvPoolToFixedDim, self).__init__()
        self.conv = nn.Conv2d(n_filters, 128, kernel_size=1)  
        self.fc = nn.Sequential(
            nn.LayerNorm(128),          # Normalization layer
            nn.Linear(128, 32),        # Fully connected layer
            nn.ReLU(),                   # Activation function
            nn.Linear(32, 1)          # Another fully connected layer
        )

    def forward(self, x):
        # x shape [B, L, L, 8]
        x = self.conv(x)  
        x = torch.mean(x, dim=(2, 3))    
        x = self.fc(x)
        return x

class ContactModel(nn.Module):
    def __init__(self, esm_model_name, input_channels, n_filters, kernel_size, n_layers, num_bins, crop_size):
        super().__init__()
        self.esm_model_head = EsmForMaskedLM.from_pretrained(esm_model_name)
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        
        # Freeze ESM model parameters
        for param in self.esm_model_head.parameters():
            param.requires_grad = False
        
        self.con_model = self._create_network(384, n_filters, kernel_size, n_layers)

        self.cross_projection_pair_1 =  MultiHeadCrossAttentionModule((48+num_bins),256,4)
        self.cross_projection_pair_2 =  MultiHeadCrossAttentionModule((48+num_bins),256,4)
        
        # Process through contrastive projection
        self.self_attention_projection_insert_cls = MultiHeadSelfAttention_with_cls(256,4)
        self.self_attention_projection_extract_cls = MultiHeadAttentionWithCLSToken(256,4)
        self.self_attention_projection = nn.Sequential(
            nn.LayerNorm(256),          # Normalization layer
            nn.Linear(256, 64),        # Fully connected layer
            nn.ReLU(),                   # Activation function
            nn.Linear(64, 1)          # Another fully connected layer
        )
        self.crop_size = crop_size
        self.num_bins = num_bins
        self.esm_mlp_z = nn.Sequential(
            nn.LayerNorm(660),          # Normalization layer
            nn.Linear(660, 128),        # Fully connected layer
            nn.ReLU(),                   # Activation function
            nn.Linear(128, 128)          # Another fully connected layer
        )
        self.esm_mlp_s = nn.Sequential(
            nn.LayerNorm(self.esm_model_head.config.hidden_size),          # Normalization layer
            nn.Linear(self.esm_model_head.config.hidden_size, self.esm_model_head.config.hidden_size//2),        # Fully connected layer
            nn.ReLU(),                   # Activation function
            nn.Linear(self.esm_model_head.config.hidden_size//2, 256)          # Another fully connected layer
        )
        self.bin_projection = nn.Sequential(
            nn.InstanceNorm2d(n_filters),          # Normalization layer
            nn.Conv2d(n_filters, n_filters//4, kernel_size=5,padding=2),        # Fully connected layer
            nn.ReLU(),                   # Activation function
            nn.Conv2d(n_filters//4, num_bins, kernel_size=1)          # Another fully connected layer
        )
        
    def _create_network(self, input_channels, n_filters, kernel_size, n_layers):
        network = nn.Sequential()
        network.add_module('initial_conv', nn.Conv2d(input_channels, n_filters, kernel_size=1, padding=0))
        network.add_module('inst_norm', nn.InstanceNorm2d(n_filters))
        network.add_module('relu', nn.ReLU())

        dilation_rate = 1

        for i in range(n_layers):
            network.add_module(f'residual_block_{i}',
                               ResidualBlock2D(n_filters, n_filters, kernel_size, dilation_rate))
            dilation_rate *= 2

            if dilation_rate > 16:
                dilation_rate = 1

        return network
    
    def forward(self, sequences, crop_size=0, true_contact = None, validation = False):
        device = next(self.parameters()).device
        crop_size_current = crop_size if crop_size != 0 else self.crop_size
        #start_time = time.time()
        #print_memory_usage()
        
        # Process in chunks to reduce memory usage if needed
        with torch.no_grad():
            # Tokenize all sequences in a batch
            inputs = self.esm_tokenizer(sequences, 
                                      return_tensors='pt', 
                                      padding='longest',
                                      max_length=crop_size_current+2,  # for cls and end tokens
                                      truncation=True)
            
            # Move inputs to the same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Get model outputs
            outputs = self.esm_model_head(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True,
                output_attentions=True
            )
            del inputs
            
            # Extract hidden states (excluding special tokens)
            hidden_states = outputs.hidden_states[-1][:,1:-1,:].detach()  # Clone to make a copy
            
            # # Process attention from all layers and reshape
            all_layers_attention = torch.stack(outputs.attentions, dim=1)[:,:,:,1:-1,1:-1].detach()
            all_layers_attention = all_layers_attention.permute(0,3,4,1,2).flatten(3,4)
            del outputs
        #print(all_layers_attention.shape)
        #print_memory_usage()
        #duration = time.time() - start_time
        #print(f"ESM2 processing time: {duration:.2f}s")
        esm_z_reshape = self.esm_mlp_z(all_layers_attention)
        del all_layers_attention
        hidden_states_projection=self.esm_mlp_s(hidden_states)
        del hidden_states
        # Compute pairwise embeddings
        #print(f"Hidden states shape: {hidden_states.shape}")
        all_pair_maps = hidden_states_projection[:,:,None,:] + hidden_states_projection[:,None,:,:]
        #print(all_pair_maps.shape)
        #print_memory_usage()
        
        # Combine pair and attention maps
        combined_maps = torch.cat([all_pair_maps, esm_z_reshape], dim=-1).permute(0,3,1,2)
        
        # Free up memory for intermediate tensors
        del esm_z_reshape
        del all_pair_maps

        # Apply MLP transformation
        
        
        #print(f"Reshaped ESM features shape: {esm_z_reshape.shape}")
        #print_memory_usage()
        
        # Process through convolutional model
        conv_output = self.con_model(combined_maps)
        bin_logits = self.bin_projection(conv_output)
        #
        bin_probs = F.softmax(bin_logits, dim=1)
        bin_probs = bin_probs.permute(0, 2, 3, 1)
        del combined_maps
        del conv_output
        if true_contact is not None:
            true_contact = true_contact.to(device)
            ##print(f"Conv output shape: {all_pair_maps.shape}")
            #print(f"Bin probs shape: {bin_probs.shape}")
            
            pair_stack = torch.cat([true_contact, bin_probs], dim=3)
            #print(f"pair_stack shape: {pair_stack.shape}")
            #print(f"hidden_states_projection shape: {hidden_states_projection.shape}")
            #pair_stack = self.con_pair(pair_stack.permute(0,3,1,2))
            #print(f"pair_stack shape: {pair_stack.shape}")
            single_stack = self.cross_projection_pair_1(hidden_states_projection, pair_stack)
            single_stack = self.cross_projection_pair_2(single_stack, pair_stack.permute(0,2,1,3))
            del hidden_states_projection,pair_stack
            single_stack = self.self_attention_projection_insert_cls(single_stack)
            
            if validation:
                B = single_stack.shape[0]  # Batch
                all_indices = torch.arange(B)
                pair_indices = torch.stack([
                    all_indices.repeat_interleave(B - 1),  # Actively reduce and repeat $B-1$ times
                    torch.cat([torch.cat((all_indices[:i], all_indices[i+1:])) for i in range(B)])  # Minuend
                ], dim=1)  # [C, 2]，C = B * (B - 1)

                # Extract feature pairs

                features_pair_1 = single_stack[pair_indices[:, 0]]  # [C, n_filters, L, L]
                features_pair_2 = single_stack[pair_indices[:, 1]]  # [C, n_filters, L, L]

                # Calculate interaction pairs, keeping only interaction_pair_1

                interaction_pair = features_pair_1 - features_pair_2 # [C, n_filters, L, L]
                # Organize interaction pairs into [B, B-1]
                del features_pair_1
                del features_pair_2
                contrastive_output = self.self_attention_projection_extract_cls(interaction_pair)
                del interaction_pair
                #print(f"Contrastive output shape: {contrastive_output.shape}")
                contrastive_output = self.self_attention_projection(contrastive_output)
                contrastive_output = F.sigmoid(contrastive_output)
                contrastive_output = contrastive_output.view(B, B - 1)  # [B, B-1]
                #print(f"Contrastive output shape: {contrastive_output.shape}")
            else:
                B = single_stack.shape[0]//2
                features_pair_1 = single_stack[:B]  
                features_pair_2 = single_stack[B:]  
                interaction_pair_1 = features_pair_1 - features_pair_2 
                interaction_pair_2 = features_pair_2 - features_pair_1 
                interaction_pair = torch.cat([interaction_pair_1, interaction_pair_2], dim=0)  
                #print(f"Interaction pair shape: {interaction_pair.shape}")
                contrastive_output = self.self_attention_projection_extract_cls(interaction_pair)
                #print(f"Contrastive output shape: {contrastive_output.shape}")
                contrastive_output = self.self_attention_projection(contrastive_output)
                #print(f"Contrastive output shape: {contrastive_output.shape}")
                contrastive_output = F.sigmoid(contrastive_output)
                #print(f"Contrastive output shape: {contrastive_output.shape}")
            return contrastive_output
        #print(f"Contrastive output shape: {contrastive_output.shape}")
        else:
            return bin_probs
        
class MultiHeadCrossAttentionModule(nn.Module):
    """Improved multi-head cross-attention module, using matrix multiplication to compute attention scores"""
    def __init__(self, contact_bins, embed_dim, num_heads):
        """
        Initialize the Multi-Head Cross Attention module

        :param contact_bins: Feature dimension of the contact map

        :param embed_dim: Feature dimension (must be a multiple of num_heads)
        :param num_heads: Number of attention heads

        """
        super(MultiHeadCrossAttentionModule, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be an integer multiple of num_heads!"
        
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads  # Dimension of each head

        # Multi-head Query, Key, and Value projections

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(contact_bins, embed_dim)
        self.value_projection = nn.Linear(contact_bins, embed_dim)

        # Output projection, used to merge multi-head attention results

        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # Add layer normalization

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, B_L_D, B_L_L_D):
        """
        :param B_L_D: Primary input tensor, with shape (B, L, D)
        :param B_L_L_D: Mapping tensor, with shape (B, L, L, D)
        :return: Output tensor, with shape (B, L, D)
        """
        residual = B_L_D

        
        # Projections to generate Q, K, and V

        Q = self.query_projection(B_L_D)  # (B, L, D)
        K = self.key_projection(B_L_L_D)  # (B, L, L, D)
        V = self.value_projection(B_L_L_D)  # (B, L, L, D)
        
        # Split Q, K, and V into multiple heads

        B, L, D = Q.shape

        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L, head_dim)
        K = K.view(B, L, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B, num_heads, L, L, head_dim)
        V = V.view(B, L, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B, num_heads, L, L, head_dim)

        # Calculate attention scores

        Q_expanded = Q.unsqueeze(-2)  # (B, num_heads, L, 1, head_dim)
        attention_scores = torch.matmul(Q_expanded, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, num_heads, L, L, L)
        attention_scores = attention_scores.squeeze(-2)  # (B, num_heads, L, L)

        # Apply softmax normalization

        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, num_heads, L, L)
        
        # Weight and aggregate V using attention weights

        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (B, num_heads, L, L, 1)
        weighted_V = attention_weights_expanded * V  # (B, num_heads, L, L, head_dim)
        context = weighted_V.sum(dim=3)  # (B, num_heads, L, head_dim)
        
        # Concatenate multi-head results and apply output projection

        context = context.transpose(1, 2).reshape(B, L, D)  # (B, L, D)
        output = self.output_projection(context)
        
        # Add residual connection and layer normalization

        output = self.layer_norm(output + residual)

        return output
    
class MultiHeadAttentionWithCLSToken(nn.Module):
    """Multi-head attention module, outputs only the updated CLS token"""
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadAttentionWithCLSToken, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be an integer multiple of num_heads"

        # Projection layer

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)

        # Output layer

        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # Add layer normalization and dropout

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        :param x: Input tensor, with shape (B, L, D)
        :return: Updated CLS token, with shape (B, D)
        """
        B, L, D = x.shape

        # Use the CLS token as Query, and other tokens as Key and Value

        cls_token = x[:, 0:1, :]  # Extract the CLS token, with shape (B, 1, D)
        other_tokens = x[:, 1:, :]  # Extract the other tokens, with shape (B, L-1, D)

        # Projections to generate Q, K, and V

        Q = self.query_projection(cls_token)  # (B, 1, D)
        K = self.key_projection(other_tokens)  # (B, L-1, D)
        V = self.value_projection(other_tokens)  # (B, L-1, D)

        # Split heads

        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 1, head_dim)
        K = K.view(B, L - 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L-1, head_dim)
        V = V.view(B, L - 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L-1, head_dim)

        # Calculate attention scores and apply scaling

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, num_heads, 1, L-1)

        # Apply softmax normalization

        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, num_heads, 1, L-1)
        attention_weights = self.dropout(attention_weights)

        # Weight Value with attention weights

        attention_output = torch.matmul(attention_weights, V)  # (B, num_heads, 1, head_dim)

        # Merge heads

        attention_output = attention_output.transpose(1, 2).contiguous().view(B, 1, D)  # (B, 1, D)

        # Output projection

        attention_output = self.output_projection(attention_output)
        attention_output = self.dropout(attention_output)

        # Residual connection and layer normalization

        cls_output = self.layer_norm(attention_output + cls_token)  # Apply residual connection to the CLS token only

        # Return the updated CLS token

        return cls_output.squeeze(1)  # (B, D)

class MultiHeadSelfAttention_with_cls(nn.Module):
    """Multi-head self-attention module with added CLS token, residual connection, and layer normalization"""
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadSelfAttention_with_cls, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be an integer multiple of num_heads"
        
        # CLS token embbedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Projection layer
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        
        # Output layer
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # Add layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Feed-forward network layer
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        :param x: Input tensor, with shape (B, L, D)
        :return: Output tensor, with shape (B, L+1, D), including the CLS token
        """
        B, L, D = x.shape
        
        # add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x_with_cls = torch.cat([cls_token, x], dim=1)  # (B, L+1, D)
        
        # Apply self-attention, but keep the original input for residual connection
        residual = x_with_cls
        
        # generate Q, K, V
        Q = self.query_projection(x_with_cls)  # (B, L+1, D)
        K = self.key_projection(x_with_cls)    # (B, L+1, D)
        V = self.value_projection(x_with_cls)  # (B, L+1, D)
        
        # split heads
        Q = Q.view(B, L+1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L+1, head_dim)
        K = K.view(B, L+1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L+1, head_dim)
        V = V.view(B, L+1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L+1, head_dim)
        
        # Calculate attention scores and apply scaling
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, num_heads, L+1, L+1)
        
        # Apply softmax normalization
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, num_heads, L+1, L+1)
        attention_weights = self.dropout(attention_weights)
        
        # Weight Value with attention weights 
        attention_output = torch.matmul(attention_weights, V)  # (B, num_heads, L+1, head_dim)
        
        # Merge heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, L+1, D)  # (B, L+1, D)
        
        # Output Projection
        attention_output = self.output_projection(attention_output)
        attention_output = self.dropout(attention_output)
        
        # First residual connection and layer normalization
        attention_output = self.layer_norm(attention_output + residual)
        
        # Second residual connection and feed-forward network
        ffn_output = self.ffn(attention_output)
        output = self.ffn_layer_norm(ffn_output + attention_output)
        
        return output

from Bio.PDB import  PDBParser
def extract_pdb_info(pdb_file):
    """Extract amino acid sequence, Cβ coordinates, and backbone coordinates from a PDB file"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    backbone_coords = []  # Store the coordinates of backbone atoms

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':  # Skip anomalous residues
                    continue

                try:
                    n = residue['N'].coord
                    ca = residue['CA'].coord
                    c = residue['C'].coord
                    backbone_coords.append([n, ca, c])
                except KeyError:
                    # Missing backbone atom(s), skip this residue
                    continue

    return np.array(backbone_coords, dtype=np.float32)
