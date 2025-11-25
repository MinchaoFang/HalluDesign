import argparse
import os.path

import json, time, os, sys, glob


sys.path.insert(0, '/home/caolongxingLab/caolongxing/DeepLearning/mpnn/github_repo/')

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import shutil
import warnings
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import os.path
from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.rosetta.core.scoring.hbonds import HBondSet,fill_hbond_set

if torch.cuda.is_available():
    print("Using GPU as backend ...")
else:
    print("No GPU device found, and using CPU as backend ...")
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def parse_args( argv ):
    argv_tmp = sys.argv
    sys.argv = argv
    description = 'do protein sequence design using the MPNN model ...'
    parser = argparse.ArgumentParser( description = description )
    parser.add_argument('-pdbs', type=str, nargs='*', help='name of the input pdb file')
    parser.add_argument('-pdb_list', type=str, help='a list file of all pdb files')
    parser.add_argument('-num_trials', type=int, default=1,  help='total number of trials for each design and only the one with the lowest mpnn score will be dumped')
    parser.add_argument("-relax_cycles", type=int, default="1", help="The number of MPNN->FastRelax cycles to perform (default 1)" )
    parser.add_argument("-omit_AAs", type=list, default='CX', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    parser.add_argument("-temperature", type=float, default=0.0001, help='An a3m file containing the MSA of your target' )
    parser.add_argument("-output_path", type=str, default='', help="the output path")
    parser.add_argument("-extra_res_fa", type=str, required=True, help="the param file of the small molecule")
    parser.add_argument("-scorefile", type=str, default='mpnn_scores.sc', help="the output score file with rosetta score per res")
    parser.add_argument('-prefix', type=str, default='', help='the prefix of the output file name')
    args = parser.parse_args()
    sys.argv = argv_tmp

    return args

args = parse_args( sys.argv )

init( "-beta_nov16 -in:file:silent_struct_type binary -output_pose_energies_table false" +
    " -holes:dalphaball /home/caolongxingLab/caolongxing/bin/DAlphaBall.gcc" +
    f" -extra_res_fa {args.extra_res_fa}" +
    " -use_terminal_residues true -mute basic.io.database core.scoring" +
    " -dunbrack_prob_buried 0.8 -dunbrack_prob_nonburied 0.8" +
    " -dunbrack_prob_buried_semi 0.8 -dunbrack_prob_nonburied_semi 0.8" )

if args.pdbs == None:
    assert (args.pdb_list != None)
    with open(args.pdb_list) as f:
        all_pdbs = [line.strip() for line in f]
else:
    all_pdbs = args.pdbs

my_sfxn = get_score_function()
opts = my_sfxn.energy_method_options()
hb_opts = opts.hbond_options()
hb_opts.decompose_bb_hb_into_pair_energies(True)
opts.hbond_options(hb_opts)
my_sfxn.set_energy_method_options(opts)

# what's the best cutoff values??????
# rosetta_score_cutoff: 0.0, dist_cutoff: 5.0, 2023-03-07, polar interactions
def get_fixed_residues(pose, rosetta_score_cutoff=-0.1, dist_cutoff = 4.0):

    fixed_residues = set()

    # with favorable rosetta energy
    my_sfxn.score(pose)
    graph = pose.energies().energy_graph()
    two_body_terms = graph.active_2b_score_types()
    ligand_ires = pose.sequence().index('Z') + 1
    for ires in range(1, pose.size()+1):
        edge = graph.find_energy_edge(ires, ligand_ires)
        if edge is None: continue
        emap = edge.fill_energy_map()
        score = 0
        for term in two_body_terms:
            score += emap[term]
        
        if score < rosetta_score_cutoff:
            fixed_residues.add(ires)


    # hbonds with his
    my_sfxn.score(pose)
    hbset = HBondSet()
    derivatives=False
    exclude_bb=True
    exclude_bsc=True
    exclude_scb=True
    exclude_sc=False
    fill_hbond_set(
        pose, derivatives, hbset, exclude_bb, exclude_bsc, exclude_scb,
        exclude_sc
    )
    info = pose.pdb_info()
    hbond_res = set()
    hbond_res.add(ligand_ires)
    num_hbond_res = -1
    while len(hbond_res)>num_hbond_res:
        num_hbond_res = len(hbond_res)

        temp_set = set()
        for resi in hbond_res:
            for bond in hbset.residue_hbonds(resi):
                temp_set.add(bond.don_res())
                temp_set.add(bond.acc_res())
        hbond_res = hbond_res | temp_set

    fixed_residues = fixed_residues | hbond_res

    # dist

    def get_residue_coords(rsd):
        rsd_coords = []
        for iatom in range(1, rsd.nheavyatoms()+1):
            xyz = rsd.xyz(iatom)
            rsd_coords.append([xyz.x, xyz.y, xyz.z])
        rsd_coords = np.array(rsd_coords)
        return rsd_coords
    ligand_coords = get_residue_coords(pose.residue(ligand_ires))

    for resi in range(1, pose.size()+1):
        rsd = pose.residue(resi)
        if rsd.type().is_ligand():
            continue
        # exclude the backbone atoms
        rsd_coords = get_residue_coords(rsd)[4:]

        dist = np.sqrt( np.sum( np.square( rsd_coords[None,...] - ligand_coords[:,None,...] ), axis=-1 ) )

        if np.any(dist<dist_cutoff):
            fixed_residues.add(resi)

    # remove the ligand ires
    fixed_residues.remove(ligand_ires)

    return list(fixed_residues)

def init_seq_optimize_model():

    # model configs
    model_weights_path = '/home/caolongxingLab/caolongxing/DeepLearning/mpnn/vanilla_model_weights/v_48_020.pt'

    hidden_dim =128
    num_layers = 3
    backbone_noise = 0.00
    num_connections = 48


    model = ProteinMPNN(num_letters=21, 
                        node_features=hidden_dim,
                        edge_features=hidden_dim,
                        hidden_dim=hidden_dim,
                        num_encoder_layers=num_layers,
                        num_decoder_layers=num_layers,
                        augment_eps=backbone_noise,
                        k_neighbors=num_connections)
    model.to(device)
    print('Number of model parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

# extract features from pose

def extract_coords_from_pose(pose, atoms=['N','CA','C'], chain=None):
  '''
  input:  x = PDB filename
          atoms = atoms to extract (optional)
  output: (length, atoms, coords=(x,y,z)), sequence
  '''

  alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
  states = len(alpha_1)
  alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
             'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']
  
  aa_1_N = {a:n for n,a in enumerate(alpha_1)}
  aa_3_N = {a:n for n,a in enumerate(alpha_3)}
  aa_N_1 = {n:a for n,a in enumerate(alpha_1)}
  aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
  aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}
  
  def AA_to_N(x):
    # ["ARND"] -> [[0,1,2,3]]
    x = np.array(x);
    if x.ndim == 0: x = x[None]
    return [[aa_1_N.get(a, states-1) for a in y] for y in x]
  
  def N_to_AA(x):
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x);
    if x.ndim == 1: x = x[None]
    return ["".join([aa_N_1.get(a,"-") for a in y]) for y in x]

  def rosetta_xyz_to_numpy(x):
    return np.array([x.x, x.y, x.z])

  xyz,seq,min_resn,max_resn = {},{},1e6,-1e6

  # the pdb info struct
  info = pose.pdb_info()

  for resi in range(1, pose.size()+1):
    # for each residue
    ch = info.chain(resi)

    if ch == chain:
        residue = pose.residue(resi)

        # exclude ligand
        # ligand has no backbone atoms
        if residue.type().is_ligand(): continue

        resn = resi
        resa,resn = "",int(resn)-1

        if resn < min_resn: 
            min_resn = resn
        if resn > max_resn: 
            max_resn = resn

        xyz[resn] = {}
        xyz[resn][resa] = {}

        seq[resn] = {}

        residue_name = residue.name3()
        seq[resn][resa] = residue_name

        # for each heavy atom
        for iatm in range(1, residue.nheavyatoms()+1):
            atom_name = residue.atom_name(iatm).strip()
            atom_xyz = rosetta_xyz_to_numpy( residue.xyz(iatm) )

            xyz[resn][resa][atom_name] = atom_xyz

  # convert to numpy arrays, fill in missing values
  seq_,xyz_ = [],[]
  try:
      for resn in range(min_resn,max_resn+1):
        if resn in seq:
          for k in sorted(seq[resn]): seq_.append(aa_3_N.get(seq[resn][k],20))
        else: seq_.append(20)
        if resn in xyz:
          for k in sorted(xyz[resn]):
            for atom in atoms:
              if atom in xyz[resn][k]: xyz_.append(xyz[resn][k][atom])
              else: xyz_.append(np.full(3,np.nan))
        else:
          for atom in atoms: xyz_.append(np.full(3,np.nan))
      return np.array(xyz_).reshape(-1,len(atoms),3), N_to_AA(np.array(seq_))
  except TypeError:
      return 'no_chain', 'no_chain'

def parse_pose(pose, use_chains = ['A','B']):
    c=0
    pdb_dict_list = []
    '''
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    '''

    chain_alphabet = use_chains

    my_dict = {}
    s = 0
    concat_seq = ''
    concat_N = []
    concat_CA = []
    concat_C = []
    concat_O = []
    concat_mask = []
    coords_dict = {}
    for letter in chain_alphabet:
        xyz, seq = extract_coords_from_pose(pose, atoms=['N','CA','C','O'], chain=letter)
        if type(xyz) != str:
            concat_seq += seq[0]
            my_dict['seq_chain_'+letter]=seq[0]
            coords_dict_chain = {}
            coords_dict_chain['N_chain_'+letter]=xyz[:,0,:].tolist()
            coords_dict_chain['CA_chain_'+letter]=xyz[:,1,:].tolist()
            coords_dict_chain['C_chain_'+letter]=xyz[:,2,:].tolist()
            coords_dict_chain['O_chain_'+letter]=xyz[:,3,:].tolist()
            my_dict['coords_chain_'+letter]=coords_dict_chain
            s += 1
    my_dict['name']=pose.pdb_info().name()
    my_dict['num_of_chains'] = s
    my_dict['seq'] = concat_seq
    if s <= len(chain_alphabet):
        pdb_dict_list.append(my_dict)
        c+=1

    return pdb_dict_list

def mpnn_design( model, pose ):

    # global settings
    omit_AAs_list = args.omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    bias_AAs_np = np.zeros(len(alphabet))

    pdb_dict_list = parse_pose(pose, 'A')
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=20000)

    fixed_residues = get_fixed_residues(pose)
    print("Fixed residues ", '+'.join([str(ii) for ii in fixed_residues]))
    masked_positions = []
    for ires in range(1, pose.size()+1):
        if ires in fixed_residues:
            masked_positions.append(ires)

    best_score = 999999.00
    best_seq = ''

    with torch.no_grad():

        for itry in range(args.num_trials):

            batch_clones = [copy.deepcopy(dataset_valid[0]) for i in range(1)]
            fixed_positions_dict = { batch_clones[0]['name'] : { "A" : masked_positions } }
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, None, fixed_positions_dict, None, None, None, None)

            pssm_log_odds_mask = (pssm_log_odds_all > 0.0).float() #1.0 for true, 0.0 for false

            randn_2 = torch.randn(chain_M.shape, device=X.device)
            sample_dict = model.sample(X, randn_2, S, chain_M, chain_encoding_all, residue_idx, mask=mask, temperature=args.temperature, omit_AAs_np=omit_AAs_np, bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos, omit_AA_mask=omit_AA_mask, pssm_coef=pssm_coef, pssm_bias=pssm_bias, pssm_multi=0.0, pssm_log_odds_flag=False, pssm_log_odds_mask=pssm_log_odds_mask, pssm_bias_flag=False, bias_by_res=bias_by_res_all)
            S_sample = sample_dict["S"] 

            log_probs = model(X, S, mask, chain_encoding_all, chain_M*chain_M_pos, residue_idx, randn_2)

            mask_for_loss = mask*chain_M*chain_M_pos
            scores = _scores(S_sample, log_probs, mask_for_loss)
            scores = scores.cpu().data.numpy()
            masked_chain_length_list = masked_chain_length_list_list[0]
            masked_list = masked_list_list[0]
            seq = _S_to_seq(S_sample[0], chain_M[0])
            score = scores[0]

            if score < best_score:
                best_score = score
                best_seq = seq
                print(f"iTry_{itry}: Found a better seq {best_seq} with the mpnn_score {best_score:.2f}")

    return best_seq, best_score, masked_positions

def thread_seq( pose, seq, masked_positions ):
    rsd_set = pose.residue_type_set_for_pose( core.chemical.FULL_ATOM_t )

    aa1to3=dict({'A':'ALA', 'C':'CYS', 'D':'ASP', 'E':'GLU', 'F':'PHE', 'G':'GLY',
        'H':'HIS', 'I':'ILE', 'K':'LYS', 'L':'LEU', 'M':'MET', 'N':'ASN', 'P':'PRO',
        'Q':'GLN', 'R':'ARG', 'S':'SER', 'T':'THR', 'V':'VAL', 'W':'TRP', 'Y':'TYR'})

    for resi, mut_to in enumerate( seq, start=1 ):
        if pose.residue(resi).type().is_ligand(): continue
        if masked_positions is not None and resi in masked_positions: continue
        name3 = aa1to3[ mut_to ]
        new_res = core.conformation.ResidueFactory.create_residue( rsd_set.name_map( name3 ) )
        pose.replace_residue( resi, new_res, True )
    
    return pose



# rosetta movers and score function
#xml = "/home/caolx/DeepLearning/mpnn/bakerlab_scripts/predictor_nodesign.xml"
xml = "/home/caolongxingLab/caolongxing/DeepLearning/mpnn/bakerlab_scripts/LIG_fastdesign.xml"
objs = protocols.rosetta_scripts.XmlObjects.create_from_file( xml )
# Load the movers we will need
#if True:
#    pack_monomer = objs.get_mover( 'pack_monomer' )
#else:
#    pack_monomer = objs.get_mover( 'soft_pack' )
relax_monomer = objs.get_mover( 'fastrelax' )
sfxn = core.scoring.ScoreFunctionFactory.create_score_function("beta_nov16")

# the deep learning model
model = init_seq_optimize_model()

scores = []
scores.append("description score_per_res mpnn_score")

for pdb in all_pdbs:

    pose = pose_from_pdb(pdb)

    if pdb.endswith('.pdb'): tag = pdb.split('/')[-1][:-4]
    elif pdb.endswith('.pdb.gz'): tag = pdb.split('/')[-1][:-7]
    else: continue

    for cycle in range(args.relax_cycles):
        new_seq, mpnn_score, masked_positions = mpnn_design(model, pose)
        thread_seq(pose, new_seq, masked_positions)
        relax_monomer.apply(pose)

    rosetta_sc = sfxn.score(pose) / pose.size()
    scores.append(f"{tag} {rosetta_sc:.2f} {mpnn_score:.2f}")

    output_name = os.path.join(args.output_path, f"{args.prefix}{tag}.pdb.gz")
    pose.dump_pdb(output_name)

output_scorefile = os.path.join(args.output_path, args.scorefile)
with open(output_scorefile, 'w') as f:
    for ii in scores:
        f.write(ii+'\n')
