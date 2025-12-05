import os
from data.op_utility import read_protein_info_from_copied_file, filter_protein_info,residue_to_str,smiles_to_mmcif,cdr_process,find_all_sequence_residues_in_pdb, random_protein_sequence
from eval.evaluation import run_mpnn_evaluation, self_consistency_af3, get_fake_msa ,process_confidence_metrics,process_confidence_metrics_protenix,self_consistency_protenix,read_pdb_to_atom_array,extract_coordinates
import copy
from typing import Dict, List, Tuple
import shutil
from eval.eval_utility import get_random_seeds,get_global_residue_index
from data.utility import *
import json
import re
from scripts.input_pkl_preprocess import process_single_file
def af3_op_af3_eval(pdb_file: str, 
                      cycle: int,
                      output_dir: str,
                      template_path: str,
                      template_for_eval,
                      mpnn_model,
                      mpnn_config_dict,
                      AF3Designer_model,
                      ref_time_steps,
                      num_samples,
                      num_seeds,
                      template_plddt_threshold,
                      template_plddt_threshold_length,
                      fake_msa,
                      ref_eval,
                      chain_types,
                      fixed_chains,
                      fixed_residues,
                      bais_per_residues,
                      metrics,
                      symmetry_residues,
                      symmetry_chains,
                      sm ,
                      ccd, 
                      dna,
                      rna,
                      cdr,
                      random_init,
                      framework_seq,
                      evaluator,
                      design_begin,
                      chain_number_list_cdr,
                      cyclic,
                      replace_MSA,
                      ptm,
                      run_af3: bool = True) -> Dict:
    """single pdb single cycle"""
    
    try:

        target_dir = os.path.join(output_dir, f"recycle_{cycle+1}")
        os.makedirs(target_dir, exist_ok=True)

        source_file = copy.deepcopy(pdb_file)
        
        metrics["cycle"] =cycle
        
        pdb_file = os.path.basename(pdb_file).lower().replace(".pdb","")
        pdb_file = re.sub(r"_recycle_.*_model", "", pdb_file)
        metrics['file_name'] = pdb_file
        copied_file = os.path.join(target_dir, metrics['file_name']+f"_recycle_{cycle+1}.pdb")
        chain_number_list = []
        if cdr and cycle==0:
            chain_number_list = cdr_process(cdr, source_file, copied_file)

        elif cdr and cycle>0:
            chain_number_list = chain_number_list_cdr
            shutil.copy(source_file, copied_file)
        else:
            shutil.copy(source_file, copied_file)
        chain_number_list_cdr = copy.deepcopy(chain_number_list)
        metrics["oringin_path"] = os.path.join(output_dir, "recycle_1", metrics['file_name']+"_recycle_1.pdb")
        # for fixed chains and residues, fixed res will be recognized as plldt_good res to fix
        plddt_good_indicates = []
        plddt_atoms_indicates = []
        print(chain_number_list)
        protein_info = read_protein_info_from_copied_file(copied_file)
        if framework_seq:
            print(f"framework_seq {framework_seq}")
            framework_to_fix = find_all_sequence_residues_in_pdb(copied_file,framework_seq)
            fixed_residues = framework_to_fix
        filtered_info = filter_protein_info(protein_info, fixed_chains, fixed_residues,chain_number_list)

        plddt_good_indicates = [
            f"{chain}{residue_to_str(res.id)}"
            for chain, residues in filtered_info.items()
            for res in residues
        ]
        # for symmetry chains and res design
        if symmetry_chains:
            symmetry_residues = generate_cross_chain_symmetry(protein_info, symmetry_chains)

        tag_pre = f"{metrics['file_name']}_recycle_{cycle}".lower()
        # for msa and templates embedding recycle
        previous_dir = os.path.join(output_dir, f"recycle_{cycle}")
        cif_pre_path= os.path.join(previous_dir, tag_pre.replace(".pdb", ""), 
                                  tag_pre.replace(".pdb", "") + "_model.cif")
        
        pocket_res = []
        if  sm  or dna or rna:
            pocket_res =  find_pocket_residues_based_on_distance(pdbfile=copied_file ,cutoff=8.0)
            print(len(pocket_res))

        if template_plddt_threshold > 0 and cycle >= 1:
            template_to_json = template_process(cif_pre_path, template_plddt_threshold,template_plddt_threshold_length)
            
            for result in template_to_json:
                chain_id = result['chain']
                indices = result['queryIndices']
                print(chain_id)
                print(indices)
                converted = [f"{chain_id}{index + 1}" for index in indices]  # Adjust indices (+1)
                # pockets res fix should has two requiments: plddt good and has paired-wise energy between LIG
                
                plddt_good_indicates +=  converted
                result['queryIndices'] = [int(indic[1:])-1 for indic in plddt_good_indicates if indic[0]==chain_id]
                result['templateIndices'] = result['queryIndices']

        paired = []
        print("plddt_good_protein_indicates", plddt_good_indicates)
        
        metrics["plddt_good_indicates_len"] = len(plddt_good_indicates) 
        mpnn_dir=copied_file.replace(".pdb","_mpnn_eval")
        ccd_LG = None

        metrics = run_mpnn_evaluation(copied_file,
                                      mpnn_model,
                                    mpnn_config_dict,
                                    plddt_good_indicates,
                                    pocket_res,
                                    mpnn_dir,
                                    symmetry_residues,
                                    metrics,
                                    cyclic,
                                    cycle,
                                    evaluator,
                                    bais_per_residues)
        print(f"design begin {design_begin}")       
        if design_begin:
            print("af3 evaluation")
            AF3_sm_dir=copied_file.replace(".pdb","af3_eval")
            os.makedirs(AF3_sm_dir, exist_ok=True)
            if template_for_eval:
                template_path_for_eval = template_for_eval
            else:
                template_path_for_eval = template_path
            metrics=self_consistency_af3(
                copied_file,
                AF3Designer_model,
                AF3_sm_dir,
                template_path_for_eval,
                sm,
                ccd_LG,
                ccd,
                dna,
                rna,
                ref_eval,
                chain_types,
                pocket_res,
                fixed_chains,
                plddt_good_indicates,
                cyclic,
                replace_MSA,
                metrics
            )
        else:
            print("no af3 prediction")

        # for last cycle no optimzie
        if not run_af3:
            return metrics, copied_file, chain_number_list_cdr

        # prepare for AF3
        with open(template_path, 'r') as f:
            template = json.load(f)
        tag = f"{pdb_file}_recycle_{cycle+1}"
        json_path = os.path.join(target_dir, metrics[0]['file_name']+".json")
        copied_file = metrics[0]["packed_path"] 
        input_json = template.copy()
        input_json['name'] = tag.replace(".pdb", "")
        # ! to do should allow multi-ccd
        
        count = 0
        sm_count = 0
        rna_count = 0 
        dna_count = 0 
        protein_count = 0
        chain_labels = string.ascii_uppercase[:10] 
        for chain in chain_types:
            if chain == "protein":
                if chain_labels[protein_count] not in fixed_chains:
                    if random_init and cycle == 0:
                        input_json['sequences'][count]['protein']["sequence"] = random_protein_sequence(get_chain_sequence(copied_file,chain_labels[protein_count]),plddt_good_indicates, chain_labels[protein_count])
                    else:
                        input_json['sequences'][count]['protein']["sequence"] = get_chain_sequence(copied_file,chain_labels[protein_count])
                    print(input_json['sequences'][count]['protein']["sequence"])
                protein_count +=1
            if chain == 'ligand':
                if ccd:
                    input_json['sequences'][count]["ligand"]["ccdCodes"] = [ccd[sm_count]]
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
            
        count_tuple = [count,protein_count,sm_count,rna_count,dna_count]
        input_json["modelSeeds"] = get_random_seeds(num_seeds)
        if replace_MSA:
            #msa=f'>query\n{input_json["sequences"][chain]["protein"]["sequence"]}\n'
            #originMSA = input_json['sequences'][chain]['protein']["unpairedMsa"]
#
            #input_json['sequences'][chain]['protein']["unpairedMsa"] = msa
#
            ## get old MSA
            origin_msa = input_json['sequences'][0]['protein']['unpairedMsa'].strip().splitlines()

            # new query 
            new_query_seq = input_json['sequences'][0]['protein']['sequence']

            msa_lines = []
            i = 0
            while i < len(origin_msa):
                if origin_msa[i].startswith('>'):
                    if i == 0:  # replace first MSA
                        msa_lines.append('>query')
                        msa_lines.append(new_query_seq)
                        i += 1
                        while i < len(origin_msa) and not origin_msa[i].startswith('>'):
                            i += 1
                        continue
                    else:
                        msa_lines.append(origin_msa[i])
                else:
                    msa_lines.append(origin_msa[i])
                i += 1

            # update input_json
            input_json['sequences'][0]['protein']['unpairedMsa'] = '\n'.join(msa_lines) + '\n'

        if fake_msa and cycle >=1:
            seeds_path = os.path.join(previous_dir, tag_pre.replace(".pdb", ""))
            subfolder = os.path.join(seeds_path, "all_samples")
            cif_files = [
                os.path.abspath(os.path.join(subfolder, f))  
                for f in os.listdir(subfolder)  
                if os.path.isfile(os.path.join(subfolder, f)) and f.endswith(".cif")  

            ]
            for chain in range(0,protein_count):
                if  chain_labels[chain] not in fixed_chains:
                    msa=f'>query\n{input_json["sequences"][chain]["protein"]["sequence"]}\n'
                    for cif_file in cif_files:
                        to_fake_msa_file = os.path.join(seeds_path, subfolder,subfolder+"_model.pdb")
                        convert_cif_to_pdb(cif_file,to_fake_msa_file)
                        msa=msa+get_fake_msa(to_fake_msa_file,fake_msa,chain, mpnn_model)
                    input_json['sequences'][chain]['protein']["unpairedMsa"] = msa
            for cif_file in cif_files:
                os.remove(cif_file)
        if template_plddt_threshold > 0 and cycle >=1:
            chain_id_path = split_cif_by_chain(cif_pre_path, os.path.join(previous_dir, tag_pre.replace(".pdb", "")),fixed_chains)
            _to_count = 0
            for path in chain_id_path:
                template_to_json[_to_count]["mmcif"] = read_file(pathlib.Path(path),pathlib.Path(json_path))
                del template_to_json[_to_count]["chain"]
                input_json['sequences'][_to_count]['protein']['templates'] = [template_to_json[_to_count]]
                _to_count += 1
        if paired:
            input_json["bondedAtomPairs"] = paired
        with open(json_path, 'w') as f:
            json.dump(input_json, f, indent=2)
        
        # pkl_process
        insert = None
        if ptm:
            print(f"ptm {ptm}")
            if ptm[1] =="P":
                global_index, chain_name, residue_info = get_global_residue_index(copied_file,ptm[0])
                insert_atoms_number = 12
                insert = [global_index+1,insert_atoms_number]
                print(insert)
            else:
                insert_atoms_number = 0

        result, file_name, error = process_single_file(( copied_file, target_dir, target_dir),insert)

        # run AF3
        pkl_path = os.path.join(target_dir, f'{os.path.splitext(copied_file)[0]}.pkl')
        clear_gpu_memory()
        print(json_path,  target_dir, pkl_path, ref_time_steps, num_samples)
        if ref_time_steps == 200:
            print("pure prediction")
            results_op= AF3Designer_model.single_file_process(json_path=json_path,
                                              out_dir=target_dir,
                                            ref_pdb_path=None,
                                            ref_time_steps =200,
                                            cyclic=cyclic,
                                            num_samples=num_samples)
        elif cycle == 0 and random_init:
            results_op= AF3Designer_model.single_file_process(json_path=json_path,
                                              out_dir=target_dir,
                                            ref_pdb_path=None,
                                            ref_time_steps =200,
                                            cyclic=cyclic,
                                            num_samples=num_samples)
        else:
            results_op= AF3Designer_model.single_file_process(json_path=json_path,
                                              out_dir=target_dir,
                                            ref_pdb_path=pkl_path,
                                            ref_time_steps =ref_time_steps,
                                            cyclic=cyclic,
                                            num_samples=num_samples)
        
        if results_op:
            # get output path
            tag =f"{tag}".lower()
            cif_path = os.path.join(target_dir, tag.replace(".pdb", ""), 
                                  tag.replace(".pdb", "") + "_model.cif")
            
            metrics = process_confidence_metrics(results_op,
                                                     cif_path, copied_file,
                                                     metrics[0]["oringin_path"],
                                                     metrics,
                                                     pocket_res,
                                                     chain_types,
                                                     fixed_chains,
                                                     count_tuple)
            pdb_output = cif_path.replace(".cif", ".pdb")
            # convert cif to pdb
            if convert_cif_to_pdb(cif_path, pdb_output):
                return metrics, pdb_output,chain_number_list_cdr
            else:
                print(f"convert cif to pdb failed, keep old file to go on")
                return metrics, copied_file,chain_number_list_cdr
        else:
            print(f"AF3 failed, keep old file to go on")
            metrics['AF3_Status'] = 'Failed'
            return metrics, copied_file,chain_number_list_cdr
            
    except Exception as e:
        print(f"file error: {str(e)}")
        return metrics, copied_file,chain_number_list_cdr
    

# for Protenix import
try:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))    
    sys.path.insert(0,os.path.join(current_dir,"Protenix"))
    from Protenix.runner.inference import ProtenixInferrer

    def protenix_op_protenix_eval(pdb_file: str, 
                      cycle: int,
                      output_dir: str,
                      template_path: str,
                      template_for_eval,
                      mpnn_model,
                      mpnn_config_dict,
                      Designer_model,
                      ref_time_steps,
                      num_samples,
                      num_seeds,
                      ref_eval,
                      chain_types,
                      fixed_chains,
                      fixed_residues,
                      bais_per_residues,
                      metrics,
                      ptm,
                      symmetry_residues,
                      symmetry_chains,
                      sm ,
                      ccd,  
                      dna,
                      rna,
                      cdr,
                      framework_seq,
                      evaluator,
                      design_begin,
                      chain_number_list_cdr,
                      cyclic,
                      random_init,
                      run_af3: bool = True) -> Dict:
        """Process a single PDB file in one iteration"""

        try:

            target_dir = os.path.join(output_dir, f"recycle_{cycle+1}")
            os.makedirs(target_dir, exist_ok=True)

            source_file = copy.deepcopy(pdb_file)
            metrics["cycle"] =cycle
            print(pdb_file)
            pdb_file = os.path.basename(pdb_file).lower().replace(".pdb","")
            pdb_file = re.sub(r"_recycle_.*", "", pdb_file)
            metrics['file_name'] = pdb_file
            copied_file = os.path.join(target_dir, metrics['file_name']+f"_recycle_{cycle+1}.pdb")
            chain_number_list = []
            if cdr and cycle==0:
                chain_number_list = cdr_process(cdr, source_file, copied_file)

            elif cdr and cycle>0:
                chain_number_list = chain_number_list_cdr
                shutil.copy(source_file, copied_file)
            else:
                shutil.copy(source_file, copied_file)
            chain_number_list_cdr = copy.deepcopy(chain_number_list)
            metrics["oringin_path"] = os.path.join(output_dir, "recycle_1", metrics['file_name']+"_recycle_1.pdb")
            # for fixed chains and residues, fixed res will be recognized as plldt_good res to fix
            plddt_good_indicates = []
            plddt_atoms_indicates = []
            print(chain_number_list)
            protein_info = read_protein_info_from_copied_file(copied_file)
            if framework_seq:
                print(f"framework_seq {framework_seq}")
                framework_to_fix = find_all_sequence_residues_in_pdb(copied_file,framework_seq)
                fixed_residues = framework_to_fix
            print(fixed_residues)
            filtered_info = filter_protein_info(protein_info, fixed_chains, fixed_residues,chain_number_list)

            plddt_good_indicates = [
                f"{chain}{residue_to_str(res.id)}"
                for chain, residues in filtered_info.items()
                for res in residues
            ]
            # for symmetry chains and res design
            if symmetry_chains:
                symmetry_residues = generate_cross_chain_symmetry(protein_info, symmetry_chains)

            tag_pre = f"{metrics['file_name']}_recycle_{cycle}".lower()
            # for msa and templates embedding recycle
            previous_dir = os.path.join(output_dir, f"recycle_{cycle}")
            cif_pre_path = os.path.join(previous_dir,tag_pre,"seed_123","predictions",f"{tag_pre}_seed_123_sample_0.cif")

            pocket_res = []
            if  sm  or dna or rna:
                pocket_res =  find_pocket_residues_based_on_distance(pdbfile=copied_file ,cutoff=8.0)
                print(len(pocket_res))

            paired = []
            print("plddt_good_protein_indicates", plddt_good_indicates)

            metrics["plddt_good_indicates_len"] = len(plddt_good_indicates) 
            mpnn_dir=copied_file.replace(".pdb","_mpnn_eval")
            ccd_codes =  None
            ccd_LG = None
            if ccd and sm:
                ccd_LG=smiles_to_mmcif(sm[0], molecule_id='LIG1',molecule_name='LIG1')

                print(ccd_LG)
                ccd_codes=["LIG1"]
            metrics = run_mpnn_evaluation(copied_file,
                                          mpnn_model,
                                        mpnn_config_dict,
                                        plddt_good_indicates,
                                        pocket_res,
                                        mpnn_dir,
                                        symmetry_residues,
                                        metrics,
                                        cyclic,
                                        cycle,
                                        evaluator,
                                        bais_per_residues)
            print(f"design begin {design_begin}")       
            if design_begin:
                print("protenix evaluation")
                AF3_sm_dir=copied_file.replace(".pdb","af3_eval")
                os.makedirs(AF3_sm_dir, exist_ok=True)
                if template_for_eval:
                    template_path_for_eval = template_for_eval
                else:
                    template_path_for_eval = template_path
                metrics = self_consistency_protenix(
                    scaffold_path=copied_file,
                    AF3Designer_model=Designer_model,
                    output_dir=AF3_sm_dir,
                    template_path=template_path_for_eval,
                    sm=sm,
                    ccd_LG=ccd_LG,
                    ccd_codes=ccd_codes,
                    dna=dna,
                    rna=rna,
                    ref_eval=ref_eval,
                    chain_types=chain_types,
                    pocket_res=pocket_res,
                    fixed_chains=fixed_chains,
                    plddt_good_indicates=plddt_good_indicates,
                    cyclic=cyclic,
                    metrics=metrics,
                    random_init=random_init
                )

            else:
                print("no af3 prediction")

            # for last cycle no optimzie
            if not run_af3:
                return metrics, copied_file, chain_number_list_cdr

            with open(template_path, 'r') as f:
                template = json.load(f)
            tag = f"{pdb_file}_recycle_{cycle+1}"
            json_path = os.path.join(target_dir, metrics[0]['file_name']+".json")
            copied_file = metrics[0]["packed_path"] 
            input_json = template.copy()
            if isinstance(Designer_model, ProtenixInferrer):
                print("Designer_model is ProtenixInferrer.")
                # Prepare to run Protenix

                input_json[0]['name'] = tag.replace(".pdb", "")
                # ! to do should allow multi-ccd

                count = 0
                sm_count = 0
                rna_count = 0 
                dna_count = 0 
                protein_count = 0
                chain_labels = string.ascii_uppercase[:10] 
                for chain in chain_types:
                    print(chain)
                    if chain == 'protein':
                        if chain_labels[protein_count] not in fixed_chains:
                            if random_init and cycle == 0:
                                print(get_chain_sequence(copied_file,chain_labels[protein_count]))
                                input_json[0]['sequences'][count]['proteinChain']["sequence"]  = random_protein_sequence(get_chain_sequence(copied_file,chain_labels[protein_count]),plddt_good_indicates, chain_labels[protein_count])
                            else:
                                input_json[0]['sequences'][count]['proteinChain']["sequence"] = get_chain_sequence(copied_file,chain_labels[protein_count])
                            protein_count +=1
                            print(input_json[0]['sequences'][count]['proteinChain']["sequence"])

                    if chain == 'ligand':
                        input_json[0]['sequences'][count]["ligand"]["ligand"] = sm[sm_count]
                        sm_count +=1

                    if chain == 'dna':
                        input_json[0]['sequences'][count]["dnaSequence"]["sequence"] = dna[sm_count]
                        dna_count +=1
                    print(rna)
                    if chain == 'rna':
                        input_json[0]['sequences'][count]["rnaSequence"]["sequence"] = rna[rna_count]
                        print(rna[rna_count])
                        rna_count +=1
                    count += 1
                count_tuple = [count,protein_count,sm_count,rna_count,dna_count]

                if paired:
                    input_json["bondedAtomPairs"] = paired
                with open(json_path, 'w') as f:
                    json.dump(input_json, f, indent=2)

                # pkl_process
                pkl_path = os.path.join(target_dir, f'{os.path.splitext(copied_file)[0]}.pkl')
                atom_array = read_pdb_to_atom_array(copied_file)
                pred_coordinates_tensor = extract_coordinates(atom_array, as_tensor=True)
                print(pred_coordinates_tensor.shape)
                #if ptm:
                #    num_points = 10
                #    residues = ptm.split()
                #    chain = residues[0][0]  
                #    residue_ids = [int(res[1:-1]) for res in residues]  
                #    gaussian_std_dev = 2.0 # Adjust this to make the Gaussian more spread out or concentrated
#   
                #    # Step 3: Generate Gaussian sampled points
                #    gaussian_points = generate_gaussian_from_residues(
                #        atom_array,
                #        chain,
                #        residue_ids,
                #        num_points,
                #        std_dev=gaussian_std_dev,
                #        as_tensor=True
                #    )
                #    print(gaussian_points.shape)

                torch.save(pred_coordinates_tensor,pkl_path)
                if cycle == 0 and random_init:
                    results_op=Designer_model.predict(
                    input_json_path=json_path,
                    dump_dir=target_dir,
                    seed=123
                    )
                elif ref_time_steps == 200:
                    print("pure prediction")
                    results_op=Designer_model.predict(
                    input_json_path=json_path,
                    dump_dir=target_dir,
                    seed=123
                    )
                else:
                    print(json_path, pkl_path)
                    results_op=Designer_model.predict(
                    input_json_path=json_path,
                    dump_dir=target_dir,
                    seed=123,
                    input_atom_array_path=pkl_path,
                    diffusion_steps = ref_time_steps
                    )

                if results_op:
                    # get Protenix output path
                    cif_path = os.path.join(target_dir,tag,"seed_123","predictions",f"{tag}_seed_123_sample_0.cif")
                    print(cif_path)
                    metrics = process_confidence_metrics_protenix(results_op,
                                                             cif_path, copied_file,
                                                             metrics[0]["oringin_path"],
                                                             metrics,
                                                             pocket_res,
                                                             chain_types,
                                                             fixed_chains,
                                                             count_tuple)
                    pdb_output = cif_path.replace(".cif", ".pdb")
                    print(pdb_output)
                    # Convert CIF to PDB for the next iteration

                    if convert_cif_to_pdb(cif_path, pdb_output):
                        return metrics, pdb_output,chain_number_list_cdr
                    else:
                        print("CIF to PDB conversion failed; continuing with the original file")
                        return metrics, copied_file,chain_number_list_cdr

                else:
                    print("Protenix processing failed; continuing with the original file")
                    metrics['AF3_Status'] = 'Failed'
                    return metrics, copied_file,chain_number_list_cdr

        except Exception as e:
            print(f"Failed to process the file: {str(e)}")
            return metrics, copied_file,chain_number_list_cdr
except Exception as e:
    print(f"unable to import Protenix modules: {str(e)}")
        