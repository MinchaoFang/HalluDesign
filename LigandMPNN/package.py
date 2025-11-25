
import argparse
import copy
import json
import os.path
import random
import sys

import numpy as np
import torch
import sys
sys.path.append(os.path.dirname(p=os.path.abspath(__file__)))

from data_utils import (
    alphabet,
    element_dict_rev,
    featurize,
    get_score,
    get_seq_rec,
    parse_PDB,
    restype_1to3,
    restype_int_to_str,
    restype_str_to_int,
    write_full_PDB,
)
from model_utils import ProteinMPNN
from prody import writePDB
from sc_utils import Packer, pack_side_chains


import os

import sys

import torch

import numpy as np

import random

class MPNNModel:
    def __init__(self, model_name,
                 T, 
                 ligand_mpnn_use_side_chain_context,
                 ligand_mpnn_use_atom_context,
                 number_of_packs_per_design,
                 pack_side_chains,
                 parse_atoms_with_zero_occupancy,
                 pack_with_ligand_context,
                 repack_everything):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.T =T
        self.checkpoint_path_dir = os.path.join(os.path.dirname(p=os.path.abspath(__file__)), "model_params")
        self.model_name = model_name
        self.ligand_mpnn_use_atom_context = ligand_mpnn_use_atom_context
        self.ligand_mpnn_use_side_chain_context = ligand_mpnn_use_side_chain_context
        self.parse_all_atoms_flag = self.ligand_mpnn_use_side_chain_context or (pack_side_chains and not repack_everything)
        self.model, self.model_sc, self.atom_context_num = self.init_mpnn_model()
        self.number_of_packs_per_design = number_of_packs_per_design
        self.pack_side_chains = pack_side_chains
        self.repack_everything =repack_everything
        self.parse_atoms_with_zero_occupancy = parse_atoms_with_zero_occupancy
        self.pack_with_ligand_context= pack_with_ligand_context

    def init_mpnn_model(self):
        checkpoint_path = self.get_checkpoint_path(self.model_name)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        atom_context_num, k_neighbors = self.get_model_parameters(checkpoint)
        self.atom_context_num = atom_context_num
        model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k_neighbors,
            device=self.device,
            atom_context_num=atom_context_num,
            model_type=self.model_name,  # Assuming model_type is the same as model_name

            ligand_mpnn_use_side_chain_context=self.ligand_mpnn_use_side_chain_context,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        model_sc = self.init_packer_model()

        return model, model_sc, atom_context_num

    def get_checkpoint_path(self, model_name):
        model_paths = {
            "protein_mpnn": "proteinmpnn_v_48_020.pt",
            "ligand_mpnn": "ligandmpnn_v_32_010_25.pt",
            "per_residue_label_membrane_mpnn": "per_residue_label_membrane_mpnn_v_48_020.pt",
            "global_label_membrane_mpnn": "global_label_membrane_mpnn_v_48_020.pt",
            "soluble_mpnn": "solublempnn_v_48_020.pt",
        }
        if model_name in model_paths:
            return os.path.join(self.checkpoint_path_dir, model_paths[model_name])
        else:
            print("Choose one of the available models")
            sys.exit()

    def get_model_parameters(self, checkpoint):
        if self.model_name == "ligand_mpnn":
            atom_context_num = checkpoint["atom_context_num"]
            k_neighbors = checkpoint["num_edges"]
        else:
            atom_context_num = 1

            k_neighbors = checkpoint["num_edges"]
        return atom_context_num, k_neighbors

    def init_packer_model(self):
        model_sc = Packer(
            node_features=128,
            edge_features=128,
            num_positional_embeddings=16,
            num_chain_embeddings=16,
            num_rbf=16,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            atom_context_num=16,
            lower_bound=0.0,
            upper_bound=20.0,
            top_k=32,
            dropout=0.0,
            augment_eps=0.0,
            atom37_order=False,
            device=self.device,
            num_mix=3,
        )
        checkpoint_path_sc = os.path.join(self.checkpoint_path_dir, "ligandmpnn_sc_v_32_002_16.pt")
        checkpoint_sc = torch.load(checkpoint_path_sc, map_location=self.device)
        model_sc.load_state_dict(checkpoint_sc["model_state_dict"])
        model_sc.to(self.device)
        model_sc.eval()
        return model_sc

    def single_protein_mpnn_design(self, scaffold_path, output_dir_mpnn, numbers_seqs, input_bias_AA="" ,chains_to_design="",fixed_res="", bais_per_residues = None,redesigned_residues="", pack_side_chains_or_not=True,symmetry_residues="",omit_AA_list="C", weights_str=""):
        if output_dir_mpnn == "":
            pack_side_chains_or_not = False
        elif not os.path.exists(output_dir_mpnn):
            os.makedirs(output_dir_mpnn, exist_ok=True)
        
        if fixed_res:
            fixed_residues = [item for item in fixed_res.split()]
        else:
            fixed_residues = []

        if redesigned_residues:
            redesigned_residues = [item for item in redesigned_residues.split()]
        else:
            redesigned_residues = []
            
        parse_these_chains_only_list = []
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
        [scaffold_path],
        device=self.device,
        chains=parse_these_chains_only_list,
        parse_all_atoms=self.parse_all_atoms_flag,
        parse_atoms_with_zero_occupancy=self.parse_atoms_with_zero_occupancy,
    )
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
        chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
            encoded_residues.append(tmp)
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(
            zip(list(range(len(encoded_residues))), encoded_residues)
        )
        bias_AA = torch.zeros([21], device=self.device, dtype=torch.float32)
        if input_bias_AA:
            tmp = [item.split(":") for item in input_bias_AA.split(",")]
            a1 = [b[0] for b in tmp]
            a2 = [float(b[1]) for b in tmp]
            for i, AA in enumerate(a1):
                bias_AA[restype_str_to_int[AA]] = a2[i]
        
        bias_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=self.device, dtype=torch.float32
        )
        if bais_per_residues:
            bias_dict = bais_per_residues
            for residue_name, v1 in bias_dict.items():
                if residue_name in encoded_residues:
                    i1 = encoded_residue_dict[residue_name]
                    for amino_acid, v2 in v1.items():
                        if amino_acid in alphabet:
                            j1 = restype_str_to_int[amino_acid]
                            bias_AA_per_residue[i1, j1] = v2

        fixed_positions = torch.tensor(
        [int(item not in fixed_residues) for item in encoded_residues],
        device=self.device,
        )
        redesigned_positions = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues],
            device=self.device,
        )
        self.atom_context_num
        
        if len(chains_to_design) != 0:
            chains_to_design_list = chains_to_design.split(",")
        else:
            chains_to_design_list = protein_dict["chain_letters"]
        chain_mask = torch.tensor(
            np.array(
                [
                    item in chains_to_design_list
                    for item in protein_dict["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=self.device,
        )
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask

        PDB_residues_to_be_redesigned = [
            encoded_residue_dict_rev[item]
            for item in range(protein_dict["chain_mask"].shape[0])
            if protein_dict["chain_mask"][item] == 1
        ]
        PDB_residues_to_be_fixed = [
            encoded_residue_dict_rev[item]
            for item in range(protein_dict["chain_mask"].shape[0])
            if protein_dict["chain_mask"][item] == 0
        ]
        print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
        print("These residues will be fixed: ", PDB_residues_to_be_fixed)
        # specify which residues are linked
        if symmetry_residues:
            symmetry_residues_list_of_lists = [
                x.split(",") for x in symmetry_residues.split("|")
            ]
            remapped_symmetry_residues = []
            for t_list in symmetry_residues_list_of_lists:
                tmp_list = []
                for t in t_list:
                    tmp_list.append(encoded_residue_dict[t])
                remapped_symmetry_residues.append(tmp_list)
        else:
            remapped_symmetry_residues = [[]]

        if weights_str:
            symmetry_weights = [
                [float(item) for item in x.split(",")]
                for x in weights_str.split("|")
            ]
        else:
            symmetry_weights = [[]]

        omit_AA = torch.tensor(
        np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32),
        device=self.device,)
        omit_AA_per_residue = torch.zeros(
        [len(encoded_residues), 21], device=self.device, dtype=torch.float32
    )
        name = scaffold_path[scaffold_path.rfind("/") + 1 :]
        if name[-4:] == ".pdb":
            name = name[:-4]
        with torch.no_grad():
            if "Y" in list(protein_dict):
                atom_coords = protein_dict["Y"].cpu().numpy()
                atom_types = list(protein_dict["Y_t"].cpu().numpy())
                atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                number_of_atoms_parsed = np.sum(atom_mask)
            else:
                print("No ligand atoms parsed")
                number_of_atoms_parsed = 0
                atom_types = ""
                atom_coords = []
            if number_of_atoms_parsed == 0:
                print("No ligand atoms parsed")
            elif self.model_name == "ligand_mpnn":
                print(
                    f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                )
                #for i, atom_type in enumerate(atom_types):
                #    print(
                #        f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                #    )
            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=8.0,
                use_atom_context=self.ligand_mpnn_use_atom_context,
                number_of_ligand_atoms=self.atom_context_num,
                model_type=self.model_name,
            )
            number_of_batches =1
            feature_dict["batch_size"] = numbers_seqs
            B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
            # add additional keys to the feature dictionary
            feature_dict["temperature"] = self.T
            feature_dict["bias"] = (
                (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                + bias_AA_per_residue[None]
                - 1e8 * omit_AA_per_residue[None]
            )
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights
            sampling_probs_list = []
            log_probs_list = []
            decoding_order_list = []
            S_list = []
            loss_list = []
            loss_per_residue_list = []
            loss_XY_list = []
            for _ in range(number_of_batches):
                feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                    device=self.device,
                )
                output_dict = self.model.sample(feature_dict)
                # compute confidence scores
                loss, loss_per_residue = get_score(
                    output_dict["S"],
                    output_dict["log_probs"],
                    feature_dict["mask"] * feature_dict["chain_mask"],
                )
                if self.model_name == "ligand_mpnn":
                    combined_mask = (
                        feature_dict["mask"]
                        * feature_dict["mask_XY"]
                        * feature_dict["chain_mask"]
                    )
                else:
                    combined_mask = feature_dict["mask"] * feature_dict["chain_mask"]
                loss_XY, _ = get_score(
                    output_dict["S"], output_dict["log_probs"], combined_mask
                )
                # -----
                S_list.append(output_dict["S"])
                log_probs_list.append(output_dict["log_probs"])
                sampling_probs_list.append(output_dict["sampling_probs"])
                decoding_order_list.append(output_dict["decoding_order"])
                loss_list.append(loss)
                loss_per_residue_list.append(loss_per_residue)
                loss_XY_list.append(loss_XY)
            S_stack = torch.cat(S_list, 0)
            log_probs_stack = torch.cat(log_probs_list, 0)
            sampling_probs_stack = torch.cat(sampling_probs_list, 0)
            decoding_order_stack = torch.cat(decoding_order_list, 0)
            loss_stack = torch.cat(loss_list, 0)
            loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
            loss_XY_stack = torch.cat(loss_XY_list, 0)
            rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
            rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)
            native_seq = "".join(
                [restype_int_to_str[AA] for AA in feature_dict["S"][0].cpu().numpy()]
            )
            seq_np = np.array(list(native_seq))
            seq_out_str = []
            for mask in protein_dict["mask_c"]:
                seq_out_str += list(seq_np[mask.cpu().numpy()])
                seq_out_str += [":"]
            seq_out_str = "".join(seq_out_str)[:-1]

            out_dict = {}
            out_dict["generated_sequences"] = S_stack.cpu()
            out_dict["sampling_probs"] = sampling_probs_stack.cpu()
            out_dict["log_probs"] = log_probs_stack.cpu()
            out_dict["decoding_order"] = decoding_order_stack.cpu()
            out_dict["native_sequence"] = feature_dict["S"][0].cpu()
            out_dict["mask"] = feature_dict["mask"][0].cpu()
            out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
            out_dict["seed"] = self.seed
            out_dict["temperature"] = self.T
            if pack_side_chains_or_not:
                print("Packing side chains...")
                sc_feature_dict = featurize(
                    protein_dict,
                    cutoff_for_score=8.0,
                    use_atom_context=self.pack_with_ligand_context,
                    number_of_ligand_atoms=16,
                    model_type="ligand_mpnn",
                )
                B = numbers_seqs
                for k, v in sc_feature_dict.items():
                    if k != "S":
                        try:
                            num_dim = len(v.shape)
                            if num_dim == 2:
                                sc_feature_dict[k] = v.repeat(B, 1)
                            elif num_dim == 3:
                                sc_feature_dict[k] = v.repeat(B, 1, 1)
                            elif num_dim == 4:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1)
                            elif num_dim == 5:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1, 1)
                        except:
                            pass
                X_stack_list = []
                X_m_stack_list = []
                b_factor_stack_list = []
                for _ in range(self.number_of_packs_per_design):
                    X_list = []
                    X_m_list = []
                    b_factor_list = []
                    for c in range(number_of_batches):
                        sc_feature_dict["S"] = S_list[c]
                        sc_dict = pack_side_chains(
                            sc_feature_dict,
                            self.model_sc,
                            3,
                            16,
                            self.repack_everything,
                        )
                        X_list.append(sc_dict["X"])
                        X_m_list.append(sc_dict["X_m"])
                        b_factor_list.append(sc_dict["b_factors"])
                    X_stack = torch.cat(X_list, 0)
                    X_m_stack = torch.cat(X_m_list, 0)
                    b_factor_stack = torch.cat(b_factor_list, 0)
                    X_stack_list.append(X_stack)
                    X_m_stack_list.append(X_m_stack)
                    b_factor_stack_list.append(b_factor_stack)
                    
            pdb_path_stack =[]
            sequences_stack = []
            for ix in range(S_stack.shape[0]):
                ix_suffix = ix 

                seq = "".join(
                    [restype_int_to_str[AA] for AA in S_stack[ix].cpu().numpy()]
                )

                # write full PDB files
                if pack_side_chains_or_not:
                    for c_pack in range(self.number_of_packs_per_design):
                        X_stack = X_stack_list[c_pack]
                        X_m_stack = X_m_stack_list[c_pack]
                        b_factor_stack = b_factor_stack_list[c_pack]
                        out_packed=output_dir_mpnn+"/" + name + self.model_name + "_" + str(ix_suffix) + "_" + str(c_pack + 1) + "" + ".pdb"
                        write_full_PDB(
                            out_packed,
                            X_stack[ix].cpu().numpy(),
                            X_m_stack[ix].cpu().numpy(),
                            b_factor_stack[ix].cpu().numpy(),
                            feature_dict["R_idx_original"][0].cpu().numpy(),
                            protein_dict["chain_letters"],
                            S_stack[ix].cpu().numpy(),
                            other_atoms=other_atoms,
                            icodes=icodes,
                            force_hetatm=0,
                        )
                        pdb_path_stack.append(out_packed)
                # -----
                # write fasta lines
                seq_np = np.array(list(seq))
                seq_out_str = []
                for mask in protein_dict["mask_c"]:
                    seq_out_str += list(seq_np[mask.cpu().numpy()])
                    seq_out_str += [":"]
                seq_out_str = "".join(seq_out_str)[:-1]
                sequences_stack.append(seq_out_str)
                        
        return sequences_stack,pdb_path_stack
    
    def duel_protein_mpnn_design(self, scaffold_path, output_dir_mpnn, numbers_seqs,  chains_to_design="",fixed_res="", redesigned_residues="", pack_side_chains_or_not=True,symmetry_residues="",omit_AA_list="C", weights_str="",parse_these_chains_only_list_input=[]):
        if output_dir_mpnn == "":
            pack_side_chains_or_not = False
        elif not os.path.exists(output_dir_mpnn):
            os.makedirs(output_dir_mpnn, exist_ok=True)
        
        if fixed_res:
            fixed_residues = [item for item in fixed_res.split()]
        else:
            fixed_residues = []

        if redesigned_residues:
            redesigned_residues = [item for item in redesigned_residues.split()]
        else:
            redesigned_residues = []
        parse_these_chains_only_list = []
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
        [scaffold_path],
        device=self.device,
        chains=parse_these_chains_only_list,
        parse_all_atoms=self.parse_all_atoms_flag,
        parse_atoms_with_zero_occupancy=self.parse_atoms_with_zero_occupancy,
    )   
        protein_dict_no_lig, backbone, other_atoms_no_lig, icodes_no_lig, _ = parse_PDB(
        [scaffold_path],
        device=self.device,
        chains=parse_these_chains_only_list_input,
        parse_all_atoms=self.parse_all_atoms_flag,
        parse_atoms_with_zero_occupancy=self.parse_atoms_with_zero_occupancy,
    )   

        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
        chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
        
        R_idx_list_no_lig = list(protein_dict_no_lig["R_idx"].cpu().numpy())  # residue indices
        chain_letters_list_no_lig = list(protein_dict_no_lig["chain_letters"])  # chain letters
        
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
            encoded_residues.append(tmp)

        encoded_residues_no_lig = []
        for i, R_idx_item in enumerate(R_idx_list_no_lig):
            tmp = str(chain_letters_list_no_lig[i]) + str(R_idx_item) + icodes_no_lig[i]
            encoded_residues_no_lig.append(tmp)
        
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(
            zip(list(range(len(encoded_residues))), encoded_residues)
        )
        
        encoded_residue_dict_no_lig = dict(zip(encoded_residues_no_lig, range(len(encoded_residues_no_lig))))
        encoded_residue_dict_rev_no_lig = dict(
            zip(list(range(len(encoded_residues_no_lig))), encoded_residues_no_lig)
        )
        
        bias_AA = torch.zeros([21], device=self.device, dtype=torch.float32)
        bias_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=self.device, dtype=torch.float32
        )
        fixed_positions = torch.tensor(
        [int(item not in fixed_residues) for item in encoded_residues],
        device=self.device,
        )
        redesigned_positions = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues],
            device=self.device,
        )
        
        bias_AA_per_residue_no_lig = torch.zeros(
            [len(encoded_residues_no_lig), 21], device=self.device, dtype=torch.float32
        )
        fixed_positions_no_lig = torch.tensor(
        [int(item not in fixed_residues) for item in encoded_residues_no_lig],
        device=self.device,
        )
        redesigned_positions_no_lig = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues_no_lig],
            device=self.device,
        )
        
        if len(chains_to_design) != 0:
            chains_to_design_list = chains_to_design.split(",")
            chains_to_design_list_no_lig = chains_to_design.split(",")
        else:
            chains_to_design_list = protein_dict["chain_letters"]
            chains_to_design_list_no_lig = protein_dict_no_lig["chain_letters"]
        chain_mask = torch.tensor(
            np.array(
                [
                    item in chains_to_design_list
                    for item in protein_dict["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=self.device,
        )
        chain_mask_no_lig = torch.tensor(
            np.array(
                [
                    item in chains_to_design_list_no_lig
                    for item in protein_dict_no_lig["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=self.device,
        )
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
            protein_dict_no_lig["chain_mask"] = chain_mask_no_lig * (1 - redesigned_positions_no_lig)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
            protein_dict_no_lig["chain_mask"] = chain_mask_no_lig * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask
            protein_dict_no_lig["chain_mask"] = chain_mask_no_lig
        PDB_residues_to_be_redesigned = [
            encoded_residue_dict_rev[item]
            for item in range(protein_dict["chain_mask"].shape[0])
            if protein_dict["chain_mask"][item] == 1
        ]
        PDB_residues_to_be_fixed = [
            encoded_residue_dict_rev[item]
            for item in range(protein_dict["chain_mask"].shape[0])
            if protein_dict["chain_mask"][item] == 0
        ]
        #print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
        #print("These residues will be fixed: ", PDB_residues_to_be_fixed)
        # specify which residues are linked
        if symmetry_residues:
            symmetry_residues_list_of_lists = [
                x.split(",") for x in symmetry_residues.split("|")
            ]
            remapped_symmetry_residues = []
            for t_list in symmetry_residues_list_of_lists:
                tmp_list = []
                for t in t_list:
                    tmp_list.append(encoded_residue_dict[t])
                remapped_symmetry_residues.append(tmp_list)
        else:
            remapped_symmetry_residues = [[]]
        if weights_str:
            symmetry_weights = [
                [float(item) for item in x.split(",")]
                for x in weights_str.split("|")
            ]
        else:
            symmetry_weights = [[]]
        omit_AA = torch.tensor(
        np.array([AA in omit_AA_list for AA in alphabet]).astype(np.float32),
        device=self.device,)
        omit_AA_per_residue = torch.zeros(
        [len(encoded_residues), 21], device=self.device, dtype=torch.float32
    )
        omit_AA_per_residue_no_lig = torch.zeros(
        [len(encoded_residues_no_lig), 21], device=self.device, dtype=torch.float32
    )
        name = scaffold_path[scaffold_path.rfind("/") + 1 :]
        if name[-4:] == ".pdb":
            name = name[:-4]
        with torch.no_grad():
            if "Y" in list(protein_dict):
                atom_coords = protein_dict["Y"].cpu().numpy()
                atom_types = list(protein_dict["Y_t"].cpu().numpy())
                atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                number_of_atoms_parsed = np.sum(atom_mask)
            else:
                print("No ligand atoms parsed")
                number_of_atoms_parsed = 0
                atom_types = ""
                atom_coords = []
            if number_of_atoms_parsed == 0:
                print("No ligand atoms parsed")
            elif self.model_name == "ligand_mpnn":
                print(
                    f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                )
                #for i, atom_type in enumerate(atom_types):
                #    print(
                #        f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                #    )
            if "Y" in list(protein_dict_no_lig):
                atom_coords_no_lig = protein_dict_no_lig["Y"].cpu().numpy()
                atom_types_no_lig = list(protein_dict_no_lig["Y_t"].cpu().numpy())
                atom_mask_no_lig = list(protein_dict_no_lig["Y_m"].cpu().numpy())
                number_of_atoms_parsed_no_lig = np.sum(atom_mask_no_lig)
            else:
                print("No ligand atoms parsed")
                number_of_atoms_parsed_no_lig = 0
                atom_types_no_lig = ""
                atom_coords_no_lig = []
            if number_of_atoms_parsed_no_lig == 0:
                print("No ligand atoms parsed")
            elif self.model_name == "ligand_mpnn":
                print(
                    f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed_no_lig}"
                )
                #for i, atom_type in enumerate(atom_types):
                #    print(
                #        f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                #    )
            feature_dict = featurize(
                protein_dict,
                cutoff_for_score=8.0,
                use_atom_context=self.ligand_mpnn_use_atom_context,
                number_of_ligand_atoms=self.atom_context_num,
                model_type=self.model_name,
            )
            feature_dict_no_lig = featurize(
                protein_dict_no_lig,
                cutoff_for_score=8.0,
                use_atom_context=self.ligand_mpnn_use_atom_context,
                number_of_ligand_atoms=self.atom_context_num,
                model_type=self.model_name,
            )
            number_of_batches =1
            feature_dict["batch_size"] = numbers_seqs
            feature_dict_no_lig["batch_size"] = numbers_seqs
            B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
            # add additional keys to the feature dictionary
            feature_dict["temperature"] = self.T
            feature_dict_no_lig["temperature"] = self.T
            
            feature_dict["bias"] = (
                (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                + bias_AA_per_residue[None]
                - 1e8 * omit_AA_per_residue[None]
            )
            feature_dict["symmetry_residues"] = remapped_symmetry_residues
            feature_dict["symmetry_weights"] = symmetry_weights
            
            feature_dict_no_lig["bias"] = (
                (-1e8 * omit_AA[None, None, :] + bias_AA).repeat([1, L, 1])
                + bias_AA_per_residue_no_lig[None]
                - 1e8 * omit_AA_per_residue_no_lig[None]
            )
            feature_dict_no_lig["symmetry_residues"] = remapped_symmetry_residues
            feature_dict_no_lig["symmetry_weights"] = symmetry_weights
            sampling_probs_list = []
            log_probs_list = []
            decoding_order_list = []
            S_list = []
            loss_list = []
            loss_per_residue_list = []
            loss_XY_list = []
            for _ in range(number_of_batches):
                feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                    device=self.device,
                )
                output_dict = self.model.sample(feature_dict)
                # compute confidence scores
                loss, loss_per_residue = get_score(
                    output_dict["S"],
                    output_dict["log_probs"],
                    feature_dict["mask"] * feature_dict["chain_mask"],
                )
                if self.model_name == "ligand_mpnn":
                    combined_mask = (
                        feature_dict["mask"]
                        * feature_dict["mask_XY"]
                        * feature_dict["chain_mask"]
                    )
                else:
                    combined_mask = feature_dict["mask"] * feature_dict["chain_mask"]
                loss_XY, _ = get_score(
                    output_dict["S"], output_dict["log_probs"], combined_mask
                )
                # -----
                S_list.append(output_dict["S"])
                log_probs_list.append(output_dict["log_probs"])
                sampling_probs_list.append(output_dict["sampling_probs"])
                decoding_order_list.append(output_dict["decoding_order"])
                loss_list.append(loss)
                loss_per_residue_list.append(loss_per_residue)
                loss_XY_list.append(loss_XY)
                
            sampling_probs_list_no_lig = []
            log_probs_list_no_lig = []
            decoding_order_list_no_lig = []
            S_list_no_lig = []
            loss_list_no_lig = []
            loss_per_residue_list_no_lig = []
            loss_XY_list_no_lig = []
            for _ in range(number_of_batches):
                feature_dict_no_lig["randn"] = torch.randn(
                    [feature_dict_no_lig["batch_size"], feature_dict_no_lig["mask"].shape[1]],
                    device=self.device,
                )
                output_dict_no_lig = self.model.sample(feature_dict_no_lig)
                # compute confidence scores
                loss_no_lig, loss_per_residue_no_lig = get_score(
                    output_dict_no_lig["S"],
                    output_dict_no_lig["log_probs"],
                    feature_dict_no_lig["mask"] * feature_dict_no_lig["chain_mask"],
                )
                if self.model_name == "ligand_mpnn":
                    combined_mask_no_lig = (
                        feature_dict_no_lig["mask"]
                        * feature_dict_no_lig["mask_XY"]
                        * feature_dict_no_lig["chain_mask"]
                    )
                else:
                    combined_mask_no_lig = feature_dict_no_lig["mask"] * feature_dict_no_lig["chain_mask"]
                loss_XY_no_lig, _ = get_score(
                    output_dict_no_lig["S"], output_dict_no_lig["log_probs"], combined_mask_no_lig
                )
                # -----
                S_list_no_lig.append(output_dict_no_lig["S"])
                log_probs_list_no_lig.append(output_dict_no_lig["log_probs"])
                sampling_probs_list_no_lig.append(output_dict_no_lig["sampling_probs"])
                decoding_order_list_no_lig.append(output_dict_no_lig["decoding_order"])
                loss_list_no_lig.append(loss)
                loss_per_residue_list_no_lig.append(loss_per_residue_no_lig)
                loss_XY_list_no_lig.append(loss_XY_no_lig)    
            sampling_probs_stack = torch.cat(sampling_probs_list, 0)
            sampling_probs_stack_no_lig = torch.cat(sampling_probs_list_no_lig, 0)
            sampling_probs_stack = sampling_probs_stack.mean(0) - sampling_probs_stack_no_lig.mean(0)
            #print(sampling_probs_stack.shape)
            
            S = 20 * torch.ones(L, dtype=torch.int64)

            #print(feature_dict["S"])
            for t in range(L):

                probs_t = sampling_probs_stack[t, :]  # [21]
                probs_t = torch.nn.functional.softmax(probs_t / self.T, dim=-1)

                S_t = torch.multinomial(probs_t, 1)[0]  # [1]

                S_true_t = feature_dict["S"][0][t]  # [1]
                S_t = (S_t * chain_mask[t] + S_true_t * (1.0 - chain_mask[t])).long()
                S[t] = S_t
            #print(S)
            S_stack = torch.cat([S.view(1,-1)], 0)
            log_probs_stack = torch.cat([torch.log(sampling_probs_stack)], 0)
            sampling_probs_stack = torch.cat([sampling_probs_stack], 0)
            decoding_order_stack = torch.cat([decoding_order_list[0]], 0)
            loss_stack = torch.cat(loss_list, 0)
            loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
            loss_XY_stack = torch.cat(loss_XY_list, 0)
            rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
            rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)
            native_seq = "".join(
                [restype_int_to_str[AA] for AA in feature_dict["S"][0].cpu().numpy()]
            )
            seq_np = np.array(list(native_seq))
            seq_out_str = []
            for mask in protein_dict["mask_c"]:
                seq_out_str += list(seq_np[mask.cpu().numpy()])
                seq_out_str += [":"]
            seq_out_str = "".join(seq_out_str)[:-1]
            out_dict = {}
            out_dict["generated_sequences"] = S_stack.cpu()
            out_dict["sampling_probs"] = sampling_probs_stack.cpu()
            out_dict["log_probs"] = log_probs_stack.cpu()
            out_dict["decoding_order"] = decoding_order_stack.cpu()
            out_dict["native_sequence"] = feature_dict["S"][0].cpu()
            out_dict["mask"] = feature_dict["mask"][0].cpu()
            out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
            out_dict["seed"] = self.seed
            out_dict["temperature"] = self.T
            protein_dict_mod = {}
            #print("Protein dict keys:")
            #for k,v in protein_dict.items():
            #    try:
            #        print(k,v.shape)
            #    except:
            #        print(k,v)
                
            if pack_side_chains_or_not:
                print("Packing side chains...")
                feature_dict_ = featurize(
                    protein_dict,
                    cutoff_for_score=8.0,
                    use_atom_context=self.pack_with_ligand_context,
                    number_of_ligand_atoms=16,
                    model_type="ligand_mpnn",
                )
                sc_feature_dict = copy.deepcopy(feature_dict_)
                B = 1
                for k, v in sc_feature_dict.items():
                    if k != "S":
                        try:
                            num_dim = len(v.shape)
                            if num_dim == 2:
                                sc_feature_dict[k] = v.repeat(B, 1)
                            elif num_dim == 3:
                                sc_feature_dict[k] = v.repeat(B, 1, 1)
                            elif num_dim == 4:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1)
                            elif num_dim == 5:
                                sc_feature_dict[k] = v.repeat(B, 1, 1, 1, 1)
                        except:
                            pass
                #print("Protein dict keys:")
                #for k,v in sc_feature_dict.items():
                #    try:
                #        print(k,v.shape)
                #    except:
                #        print(k,v)
                X_stack_list = []
                X_m_stack_list = []
                b_factor_stack_list = []
                for _ in range(self.number_of_packs_per_design):
                    X_list = []
                    X_m_list = []
                    b_factor_list = []
                    for c in range(number_of_batches):
                        sc_feature_dict["S"] = [S.view(1,-1)][c][0].view(1,-1)
                        #print(S_list[c].shape)
                        #print(sc_feature_dict["S"].shape)
                        sc_dict = pack_side_chains(
                            sc_feature_dict,
                            self.model_sc,
                            3,
                            16,
                            self.repack_everything,
                        )
                        X_list.append(sc_dict["X"])
                        X_m_list.append(sc_dict["X_m"])
                        b_factor_list.append(sc_dict["b_factors"])
                    X_stack = torch.cat(X_list, 0)
                    X_m_stack = torch.cat(X_m_list, 0)
                    b_factor_stack = torch.cat(b_factor_list, 0)
                    X_stack_list.append(X_stack)
                    X_m_stack_list.append(X_m_stack)
                    b_factor_stack_list.append(b_factor_stack)
                    #print(X_stack.shape)
                    
            pdb_path_stack =[]
            sequences_stack = []
            for ix in range(S_stack.shape[0]):
                ix_suffix = ix 

                seq = "".join(
                    [restype_int_to_str[AA] for AA in S_stack[ix].cpu().numpy()]
                )
                #print(seq)
                # write full PDB files
                if pack_side_chains_or_not:
                    for c_pack in range(self.number_of_packs_per_design):
                        X_stack = X_stack_list[c_pack]
                        X_m_stack = X_m_stack_list[c_pack]
                        b_factor_stack = b_factor_stack_list[c_pack]
                        out_packed=output_dir_mpnn + name + self.model_name + "_" + str(ix_suffix) + "_" + str(c_pack + 1) + "" + ".pdb"
                        write_full_PDB(
                            out_packed,
                            X_stack[ix].cpu().numpy(),
                            X_m_stack[ix].cpu().numpy(),
                            b_factor_stack[ix].cpu().numpy(),
                            feature_dict["R_idx_original"][0].cpu().numpy(),
                            protein_dict["chain_letters"],
                            S_stack[ix].cpu().numpy(),
                            other_atoms=other_atoms,
                            icodes=icodes,
                            force_hetatm=0,
                        )
                        pdb_path_stack.append(out_packed)
                # -----
                # write fasta lines
                seq_np = np.array(list(seq))
                seq_out_str = []
                for mask in protein_dict["mask_c"]:
                    seq_out_str += list(seq_np[mask.cpu().numpy()])
                    seq_out_str += [":"]
                seq_out_str = "".join(seq_out_str)[:-1]
                sequences_stack.append(seq_out_str)
                        
        return sequences_stack,pdb_path_stack

