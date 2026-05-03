import os
from data.op_utility import read_protein_info_from_copied_file, filter_protein_info,residue_to_str,cdr_process,find_all_sequence_residues_in_pdb, random_protein_sequence
from eval.evaluation import run_mpnn_evaluation, self_consistency_af3 ,process_confidence_metrics_af3,process_confidence_metrics_protenix,self_consistency_protenix,read_pdb_to_atom_array,extract_coordinates
import copy
from typing import Dict, List, Tuple
import shutil
from eval.eval_utility import get_random_seeds,get_global_residue_index
from data.utility import *
import json
import re
import pickle
from local_scripts.input_pkl_preprocess import process_single_file
def af3_op_af3_eval(pdb_file: str, 
                      cycle: int,
                      output_dir: str,
                      template_path: str,
                      template_for_eval,
                      mpnn_model,
                      mpnn_config_dict,
                      AF3Designer_model,
                      ref_time_steps,
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
                      enzyme_design,
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
        if cdr and cycle == 0:
            chain_number_list = cdr_process(cdr, source_file, copied_file)

        elif cdr and cycle > 0:
            chain_number_list = chain_number_list_cdr
            shutil.copy(source_file, copied_file)
        else:
            shutil.copy(source_file, copied_file)
        chain_number_list_cdr = copy.deepcopy(chain_number_list)
        metrics["oringin_path"] = os.path.join(output_dir, "recycle_1", metrics['file_name']+"_recycle_1.pdb")
        # for fixed chains and residues, fixed res will be recognized as fixed_residues_for_MPNN to fix
        fixed_residues_for_MPNN = []
        print(chain_number_list)
        protein_info = read_protein_info_from_copied_file(copied_file)
        if framework_seq:
            print(f"framework seq {framework_seq}, which will be fixed during design")
            framework_to_fix = find_all_sequence_residues_in_pdb(copied_file,framework_seq)
            fixed_residues = framework_to_fix
        filtered_info = filter_protein_info(protein_info, fixed_chains, fixed_residues,chain_number_list)
        # fixed_residues_for_MPNN save the all fixed residues
        fixed_residues_for_MPNN = [
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
        # we use the pocket res form the input structure to define the residues around ligand which we will use LigandMPNN to design
        if  sm  or dna or rna:
            pocket_res =  find_pocket_residues_based_on_distance(pdbfile=copied_file ,cutoff=8.0)
            print(len(pocket_res))

        print("plddt_good_protein_indicates", fixed_residues_for_MPNN)
        
        metrics["fixed_residues_for_MPNN_len"] = len(fixed_residues_for_MPNN) 
        mpnn_dir=copied_file.replace(".pdb","_mpnn_eval")


        metrics = run_mpnn_evaluation(copied_file,
                                      mpnn_model,
                                    mpnn_config_dict,
                                    fixed_residues_for_MPNN,
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
                ccd,
                dna,
                rna,
                chain_types,
                pocket_res,
                fixed_chains,
                fixed_residues_for_MPNN,
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
                        input_json['sequences'][count]['protein']["sequence"] = random_protein_sequence(get_chain_sequence(copied_file,chain_labels[protein_count]),fixed_residues_for_MPNN, chain_labels[protein_count])
                    else:
                        input_json['sequences'][count]['protein']["sequence"] = get_chain_sequence(copied_file,chain_labels[protein_count])
                    print(input_json['sequences'][count]['protein']["sequence"])
                protein_count +=1
            if chain == 'ligand':
                if enzyme_design:
                    print("enzyme design, no ligand input")
                    his_positions = [int(x[1:]) for x in fixed_residues]
                    input_json["bondedAtomPairs"][0][0][1] = his_positions[0]
                    input_json["bondedAtomPairs"][1][0][1] = his_positions[1]
                    input_json["bondedAtomPairs"][2][0][1] = his_positions[2]
                else:
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
        if enzyme_design:
            count_tuple = [2,1,1,0,0]
        input_json["modelSeeds"] = get_random_seeds(1)
        # use the previous MSA but replace the query sequence
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

        with open(json_path, 'w') as f:
            json.dump(input_json, f, indent=2)
        
        # pkl_process
        insert = None
        if ptm:
            # More PTM types are needed. We usually add a new ligand chain that is covalently bound to the protein.
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
        print(json_path,  target_dir, pkl_path, ref_time_steps)
        if ref_time_steps == 200:
            print("pure prediction")
            results_op= AF3Designer_model.single_file_process(json_path=json_path,
                                              out_dir=target_dir,
                                            ref_pdb_path=None,
                                            ref_time_steps =200,
                                            cyclic=cyclic,
                                            num_samples=5)
        elif cycle == 0 and random_init:
            print("pure prediction")
            results_op= AF3Designer_model.single_file_process(json_path=json_path,
                                              out_dir=target_dir,
                                            ref_pdb_path=None,
                                            ref_time_steps =200,
                                            cyclic=cyclic,
                                            num_samples=5)
        else:
            results_op= AF3Designer_model.single_file_process(json_path=json_path,
                                              out_dir=target_dir,
                                            ref_pdb_path=pkl_path,
                                            ref_time_steps =ref_time_steps,
                                            cyclic=cyclic,
                                            num_samples=5)
        
        if results_op:
            # get output path
            tag =f"{tag}".lower()
            cif_path = os.path.join(target_dir, tag.replace(".pdb", ""), 
                                  tag.replace(".pdb", "") + "_model.cif")
            
            metrics = process_confidence_metrics_af3(results_op,
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
            metrics['HalluDesign_Status'] = 'Failed'
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
                      chain_types,
                      fixed_chains,
                      fixed_residues,
                      bais_per_residues,
                      metrics,
                      symmetry_residues,
                      symmetry_chains,
                      sm ,
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
            # for fixed chains and residues, fixed res will be recognized as fixed_residues_for_MPNN res to fix
            fixed_residues_for_MPNN = []
            print(chain_number_list)
            protein_info = read_protein_info_from_copied_file(copied_file)
            if framework_seq:
                print(f"framework_seq {framework_seq}")
                framework_to_fix = find_all_sequence_residues_in_pdb(copied_file,framework_seq)
                fixed_residues = framework_to_fix

            filtered_info = filter_protein_info(protein_info, fixed_chains, fixed_residues,chain_number_list)

            fixed_residues_for_MPNN = [
                f"{chain}{residue_to_str(res.id)}"
                for chain, residues in filtered_info.items()
                for res in residues
            ]
            # for symmetry chains and res design
            if symmetry_chains:
                symmetry_residues = generate_cross_chain_symmetry(protein_info, symmetry_chains)

            pocket_res = []
            # we use the pocket res form the input structure to define the residues around ligand which we will use LigandMPNN to design
            if  sm  or dna or rna:
                pocket_res =  find_pocket_residues_based_on_distance(pdbfile=copied_file ,cutoff=8.0)
                print("pocket residues: "+ f"{len(pocket_res)}")

            print("fixed residues for MPNN", fixed_residues_for_MPNN)

            metrics["fixed_residues_for_MPNN_len"] = len(fixed_residues_for_MPNN) 
            mpnn_dir=copied_file.replace(".pdb","_mpnn_eval")
            metrics = run_mpnn_evaluation(copied_file,
                                          mpnn_model,
                                        mpnn_config_dict,
                                        fixed_residues_for_MPNN,
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
                Eval_protenix_dir=copied_file.replace(".pdb","protenix_eval")
                os.makedirs(Eval_protenix_dir, exist_ok=True)
                if template_for_eval:
                    template_path_for_eval = template_for_eval
                else:
                    template_path_for_eval = template_path
                metrics = self_consistency_protenix(
                    scaffold_path=copied_file,
                    AF3Designer_model=Designer_model,
                    output_dir=Eval_protenix_dir,
                    template_path=template_path_for_eval,
                    sm=sm,
                    dna=dna,
                    rna=rna,
                    chain_types=chain_types,
                    pocket_res=pocket_res,
                    fixed_chains=fixed_chains,
                    fixed_residues_for_MPNN=fixed_residues_for_MPNN,
                    cyclic=cyclic,
                    metrics=metrics,
                    random_init=random_init
                )

            else:
                print("no protenix evaluation")

            # for last cycle no optimzie
            if not run_af3:
                return metrics, copied_file, chain_number_list_cdr

            with open(template_path, 'r') as f:
                template = json.load(f)
            tag = f"{pdb_file}_recycle_{cycle+1}"
            json_path = os.path.join(target_dir, metrics[0]['file_name']+".json")
            copied_file = metrics[0]["packed_path"] 
            input_json = template.copy()
            print("no protenix evaluation")
            if Designer_model:
                print("Designer_model is ProtenixInferrer.")
                # Prepare to run Protenix
                input_json[0]['name'] = tag.replace(".pdb", "")
                print(input_json[0]['name'])
                # ! to do should allow multi-ccd
                count = 0
                sm_count = 0
                rna_count = 0 
                dna_count = 0 
                protein_count = 0
                chain_labels = string.ascii_uppercase[:10] 
                for chain in chain_types:
                    print("no protenix evaluation")
                    if chain == 'protein':
                        if chain_labels[protein_count] not in fixed_chains:
                            if random_init and cycle == 0:
                                input_json[0]['sequences'][count]['proteinChain']["sequence"]  = random_protein_sequence(get_chain_sequence(copied_file,chain_labels[protein_count]),fixed_residues_for_MPNN, chain_labels[protein_count])
                            else:
                                input_json[0]['sequences'][count]['proteinChain']["sequence"] = get_chain_sequence(copied_file,chain_labels[protein_count])
                                
                            protein_count +=1

                    if chain == 'ligand':
                        input_json[0]['sequences'][count]["ligand"]["ligand"] = sm[sm_count]
                        sm_count +=1

                    if chain == 'dna':
                        input_json[0]['sequences'][count]["dnaSequence"]["sequence"] = dna[dna_count]
                        dna_count +=1

                    if chain == 'rna':
                        input_json[0]['sequences'][count]["rnaSequence"]["sequence"] = rna[rna_count]
                        rna_count +=1
                    count += 1
                count_tuple = [count,protein_count,sm_count,rna_count,dna_count]

                with open(json_path, 'w') as f:
                    json.dump(input_json, f, indent=2)

                # pkl_process
                pkl_path = os.path.join(target_dir, f'{os.path.splitext(copied_file)[0]}.pkl')
                atom_array = read_pdb_to_atom_array(copied_file)
                pred_coordinates_tensor = extract_coordinates(atom_array, as_tensor=True)
                print(pred_coordinates_tensor.shape)

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
                    metrics = process_confidence_metrics_protenix(results_op,
                                                             cif_path, copied_file,
                                                             metrics[0]["oringin_path"],
                                                             metrics,
                                                             pocket_res,
                                                             chain_types,
                                                             fixed_chains,
                                                             count_tuple)
                    pdb_output = cif_path.replace(".cif", ".pdb")
                    # Convert CIF to PDB for the next iteration

                    if convert_cif_to_pdb(cif_path, pdb_output):
                        return metrics, pdb_output,chain_number_list_cdr
                    else:
                        print("CIF to PDB conversion failed; continuing with the original file")
                        return metrics, copied_file,chain_number_list_cdr

                else:
                    print("Protenix processing failed; continuing with the original file")
                    metrics['Protenix_Status'] = 'Failed'
                    return metrics, copied_file,chain_number_list_cdr

        except Exception as e:
            print(f"Failed to process the file: {str(e)}")
            return metrics, copied_file,chain_number_list_cdr
        
    def cross_model_op_protenix_eval(pdb_file: str, 
        cycle: int,
        output_dir: str,
        template_path: str,
        template_for_eval,
        extra_json_path,
        mpnn_model,
        mpnn_config_dict,
        Designer_model,
        ref_time_steps,
        chain_types,
        fixed_chains,
        fixed_residues,
        bais_per_residues,
        metrics,
        symmetry_residues,
        symmetry_chains,
        symmetry_segments,
        sm ,
        dna,
        rna,
        cdr,
        framework_seq,
        evaluator,
        design_begin,
        chain_number_list_cdr,
        cyclic,
        ptm,
        random_init,
        enzyme_design,
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
            # for fixed chains and residues, fixed res will be recognized as fixed_residues_for_MPNN res to fix
            fixed_residues_for_MPNN = []
            print(chain_number_list)
            protein_info = read_protein_info_from_copied_file(copied_file)
            if framework_seq:
                print(f"framework_seq {framework_seq}")
                framework_to_fix = find_all_sequence_residues_in_pdb(copied_file,framework_seq)
                fixed_residues = framework_to_fix

            filtered_info = filter_protein_info(protein_info, fixed_chains, fixed_residues,chain_number_list)

            fixed_residues_for_MPNN = [
                f"{chain}{residue_to_str(res.id)}"
                for chain, residues in filtered_info.items()
                for res in residues
            ]
            # for symmetry chains and res design
            if symmetry_chains:
                symmetry_residues = generate_cross_chain_symmetry(protein_info, symmetry_chains)

            if symmetry_segments:
                symmetry_residues = generate_A_chain_modulo_symmetry(protein_info, int(symmetry_segments))

            pocket_res = []
            # we use the pocket res form the input structure to define the residues around ligand which we will use LigandMPNN to design
            if  sm  or dna or rna:
                pocket_res =  find_pocket_residues_based_on_distance(pdbfile=copied_file ,cutoff=8.0)
                print("pocket residues: "+ f"{len(pocket_res)}")

            print("fixed residues for MPNN", fixed_residues_for_MPNN)

            metrics["fixed_residues_for_MPNN_len"] = len(fixed_residues_for_MPNN) 
            mpnn_dir=copied_file.replace(".pdb","_mpnn_eval")
            metrics = run_mpnn_evaluation(copied_file,
                                          mpnn_model,
                                        mpnn_config_dict,
                                        fixed_residues_for_MPNN,
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
                Eval_protenix_dir=copied_file.replace(".pdb","protenix_eval")
                os.makedirs(Eval_protenix_dir, exist_ok=True)
                if template_for_eval:
                    template_path_for_eval = template_for_eval
                else:
                    template_path_for_eval = template_path
                metrics = self_consistency_protenix(
                    scaffold_path=copied_file,
                    AF3Designer_model=Designer_model,
                    output_dir=Eval_protenix_dir,
                    template_path=template_path_for_eval,
                    sm=sm,
                    dna=dna,
                    rna=rna,
                    chain_types=chain_types,
                    pocket_res=pocket_res,
                    fixed_chains=fixed_chains,
                    fixed_residues_for_MPNN=fixed_residues_for_MPNN,
                    cyclic=cyclic,
                    metrics=metrics,
                    random_init=random_init
                )

            else:
                print("no protenix evaluation")

            if cycle % 2 == 0:
                # for last cycle no optimzie
                if not run_af3:
                    return metrics, copied_file, chain_number_list_cdr

                with open(template_path, 'r') as f:
                    template = json.load(f)
                tag = f"{pdb_file}_recycle_{cycle+1}"
                json_path = os.path.join(target_dir, metrics[0]['file_name']+".json")
                copied_file = metrics[0]["packed_path"] 
                input_json = template.copy()
            
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
                    if chain == 'protein':
                        if chain_labels[protein_count] not in fixed_chains:
                            if random_init and cycle == 0:
                                input_json[0]['sequences'][count]['proteinChain']["sequence"]  = random_protein_sequence(get_chain_sequence(copied_file,chain_labels[protein_count]),fixed_residues_for_MPNN, chain_labels[protein_count])
                            else:
                                input_json[0]['sequences'][count]['proteinChain']["sequence"] = get_chain_sequence(copied_file,chain_labels[protein_count])
                            if int(symmetry_segments) >= 2:
                                seq_segment = len(input_json[0]['sequences'][count]['proteinChain']["sequence"])//int(symmetry_segments)
                                input_json[0]['sequences'][count]['proteinChain']["sequence"]  = input_json[0]['sequences'][count]['proteinChain']["sequence"][:seq_segment] * int(symmetry_segments)
                            protein_count +=1

                    if chain == 'ligand':
                        if enzyme_design:
                            print("enzyme design, no ligand input")
                            his_positions = [int(x[1:]) for x in fixed_residues]
                            print(his_positions)
                            input_json[0]["covalent_bonds"][0]["position1"] = his_positions[0]
                            input_json[0]["covalent_bonds"][1]["position1"] = his_positions[1]
                            input_json[0]["covalent_bonds"][2]["position1"] = his_positions[2]
                            
                        else:
                            input_json[0]['sequences'][count]["ligand"]["ligand"] = sm[sm_count]
                        sm_count +=1

                    if chain == 'dna':
                        input_json[0]['sequences'][count]["dnaSequence"]["sequence"] = dna[dna_count]
                        dna_count +=1

                    if chain == 'rna':
                        input_json[0]['sequences'][count]["rnaSequence"]["sequence"] = rna[rna_count]
                        rna_count +=1
                    count += 1
                count_tuple = [count,protein_count,sm_count,rna_count,dna_count]

                with open(json_path, 'w') as f:
                    json.dump(input_json, f, indent=2)

                # pkl_process
                pkl_path = os.path.join(target_dir, f'{os.path.splitext(copied_file)[0]}.pkl')
                atom_array = read_pdb_to_atom_array(copied_file)
                pred_coordinates_tensor = extract_coordinates(atom_array, as_tensor=True)
                print(pred_coordinates_tensor.shape)

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
                    metrics = process_confidence_metrics_protenix(results_op,
                                                             cif_path, copied_file,
                                                             metrics[0]["oringin_path"],
                                                             metrics,
                                                             pocket_res,
                                                             chain_types,
                                                             fixed_chains,
                                                             count_tuple)
                    pdb_output = cif_path.replace(".cif", ".pdb")
                    # Convert CIF to PDB for the next iteration

                    if convert_cif_to_pdb(cif_path, pdb_output):
                        return metrics, pdb_output,chain_number_list_cdr
                    else:
                        print("CIF to PDB conversion failed; continuing with the original file")
                        return metrics, copied_file,chain_number_list_cdr
                else:
                    print("Protenix processing failed; continuing with the original file")
                    metrics['op_Status'] = 'Failed'
                    return metrics, copied_file,chain_number_list_cdr
            else:
                print("using subprocess AF3 to optmize")
                template_path = extra_json_path
                with open(template_path, 'r') as f:
                    template = json.load(f)
                tag = f"{pdb_file}_recycle_{cycle+1}"
                json_path = os.path.join(target_dir, metrics[0]['file_name']+".json")
                copied_file = metrics[0]["packed_path"] 
                input_json = template.copy()
                input_json['name'] = tag.replace(".pdb", "")
                
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
                                input_json['sequences'][count]['protein']["sequence"] = random_protein_sequence(get_chain_sequence(copied_file,chain_labels[protein_count]),fixed_residues_for_MPNN, chain_labels[protein_count])
                            else:
                                input_json['sequences'][count]['protein']["sequence"] = get_chain_sequence(copied_file,chain_labels[protein_count])
                            print(input_json['sequences'][count]['protein']["sequence"])
                        protein_count +=1
                    if chain == 'ligand':
                        if enzyme_design:
                            print("enzyme design, no ligand input")
                            his_positions = [int(x[1:]) for x in fixed_residues]
                            input_json["bondedAtomPairs"][0][0][1] = his_positions[0]
                            input_json["bondedAtomPairs"][1][0][1] = his_positions[1]
                            input_json["bondedAtomPairs"][2][0][1] = his_positions[2]
                        else:
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
                input_json["modelSeeds"] = get_random_seeds(1)

                with open(json_path, 'w') as f:
                    json.dump(input_json, f, indent=2)
                
                # pkl_process
                insert = None
                if ptm:
                    # More PTM types are needed. We usually add a new ligand chain that is covalently bound to the protein.
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
                print(json_path,  target_dir, pkl_path, ref_time_steps)
                dump_result = pkl_path.replace(".pkl", "_ref_eval_result.pkl")
                if ref_time_steps == 200:
                    print("pure prediction")
                    run_AF3_evaluation_with_ref_eval(target_dir,
                                                    json_path,
                                                    None, 
                                                    dump_result, 
                                                    ref_time_steps, 5,
                                                    cyclic, )
                elif cycle == 0 and random_init:
                    print("pure prediction")
                    run_AF3_evaluation_with_ref_eval(target_dir,
                                                    json_path,
                                                    None, 
                                                    dump_result, 
                                                    ref_time_steps, 5,
                                                    cyclic, )
                else:
                    run_AF3_evaluation_with_ref_eval(target_dir,
                                                    json_path,
                                                    pkl_path, 
                                                    dump_result, 
                                                    ref_time_steps, 5,
                                                    cyclic, )
                
                with open(dump_result, "rb") as f:
                    results_op = pickle.load(f)

                if results_op:
                    
                    # get output path
                    tag =f"{tag}".lower()
                    cif_path = os.path.join(target_dir, tag.replace(".pdb", ""), tag.replace(".pdb", ""), 
                                          tag.replace(".pdb", "") + "_model.cif")
                    
                    metrics = process_confidence_metrics_af3(results_op,
                                                             cif_path, copied_file,
                                                             metrics[0]["oringin_path"],
                                                             metrics,
                                                             pocket_res,
                                                             chain_types,
                                                             fixed_chains,
                                                             count_tuple)
                    pdb_output = cif_path.replace(".cif", ".pdb")
                    print(metrics)
                    # convert cif to pdb
                    if convert_cif_to_pdb(cif_path, pdb_output):
                        return metrics, pdb_output,chain_number_list_cdr
                    else:
                        print(f"convert cif to pdb failed, keep old file to go on")
                        return metrics, copied_file,chain_number_list_cdr
                else:
                    print(f"AF3 failed, keep old file to go on")
                    metrics['HalluDesign_Status'] = 'Failed'
                    return metrics, copied_file,chain_number_list_cdr

        except Exception as e:
            print(f"Failed to process the file: {str(e)}")
            return metrics, copied_file,chain_number_list_cdr
except Exception as e:
    print(f"unable to import Protenix modules: {str(e)}")
import subprocess
from pathlib import Path
import os

def run_AF3_evaluation_with_ref_eval(output_dir, json_path, pkl_path, dump_result,
                                     ref_time_steps, num_samples, cyclic=1):
    try:

        af3_script = Path("eval") / "af3_init.py"

        if not af3_script.exists():
            raise FileNotFoundError(f"{af3_script.resolve()} does not exist!")

        command = [
            "python",
            str(af3_script),
            f"--input_json={json_path}",
            f"--ref_time_steps={ref_time_steps}",
            f"--output_base_dir={output_dir}",
            f"--cyclic={cyclic}",
            f"--num_samples={num_samples}",
            f"--ref_pdb_path={pkl_path}",
            f"--dump_result={dump_result}"
        ]

        print("Running command:", " ".join(command))

        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=3600
        )

        result.check_returncode()
        print("Command executed successfully!")
        print("Output:", result.stdout)
        return True

    except FileNotFoundError as e:
        print("File not found:", str(e))

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

import os
import re

def parse_his_positions_from_pdb_filename(pdb_path, chain_id="A"):
    """
    parse_his_positions_from_pdb_filename 
    return ["A18", "A54", "A77"]
    """

    fname = os.path.basename(pdb_path)

    match = re.search(r"hpos_([\d_]+)", fname)
    if not match:
        raise ValueError(f"no Hpos information found in filename: {fname}")

    positions = match.group(1).split("_")[:3]
    return [f"{chain_id}{pos}" for pos in positions]