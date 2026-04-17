# Protein Monomer optmization exmample
## AF3
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH && python ./HalluDesign_run.py --HalluDesign_model af3 --input_file examples/monomer/monomer.pdb --template_path examples/monomer/template_und.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/monomer/HalluDesign_op --num_seqs 2 --num_recycles 10 --ref_time_steps 50

## pdblist as input 
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH && python ./HalluDesign_run.py --HalluDesign_model af3 --pdb_list examples/monomer/pdblist --template_path examples/monomer/template_und.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/monomer/HalluDesign_op --num_seqs 2 --num_recycles 10 --ref_time_steps 50

## Protenix
export CC=$(which gcc) && export CXX=$(which g++) && python ./HalluDesign_run.py --HalluDesign_model protenix --input_file examples/monomer/monomer.pdb --template_path examples/monomer/template_und_protenix.json  --mpnn "protein_mpnn" --output_dir $(pwd)/examples/monomer/HalluDesign_op --num_seqs 2 --num_recycles 10 --ref_time_steps 50

## Cross model
export CC=$(which gcc) && export CXX=$(which g++) && python ./HalluDesign_run.py --HalluDesign_model cross_model --input_file examples/monomer/monomer.pdb --template_path examples/monomer/template_und_protenix.json  --extra_json_path examples/monomer/template_und.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/monomer/HalluDesign_op --num_seqs 2 --num_recycles 10 --ref_time_steps 150 --design_epoch_begin 8

# Protein binder optmization exmample
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH && python ./HalluDesign_run.py --HalluDesign_model af3 --input_file examples/protein_binder/Pdl1_binder_design_5.pdb --template_path examples/protein_binder/pdl1_protein.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/protein_binder/HalluDesign_op --fix_chain_index "B" --num_seqs 2 --num_recycles 10 --ref_time_steps 50 

# Ligand binder optimization example with CCD ligand
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH && python ./HalluDesign_run.py --HalluDesign_model af3 --input_file examples/ligand_binder/protein_ligand.pdb --template_path examples/ligand_binder/template_CCD.json --mpnn "ligand_mpnn" --ccd "CMP" --output_dir $(pwd)/examples/ligand_binder/HalluDesign_op_ccd --num_seqs 2 --num_recycles 10 --ref_time_steps 50

# Ligand binder optimization example with smiles ligand
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH && python ./HalluDesign_run.py --HalluDesign_model af3 --input_file examples/ligand_binder/protein_ligand.pdb --template_path examples/ligand_binder/template_sm.json --mpnn "ligand_mpnn" --sm "C1[C@@H]2[C@H]([C@H]([C@@H](O2)N3C=NC4=C(N=CN=C43)N)O)OP(=O)(O1)O"  --output_dir $(pwd)/examples/ligand_binder/HalluDesign_op_sm --num_seqs 2 --num_recycles 10 --ref_time_steps 50

# Ligand binder optimization example with protenix model
export CC=$(which gcc) && export CXX=$(which g++) && python ./HalluDesign_run.py --HalluDesign_model protenix --input_file examples/ligand_binder/protein_ligand.pdb --template_path examples/ligand_binder/template_sm_protenix.json --mpnn "ligand_mpnn" --sm "C1[C@@H]2[C@H]([C@H]([C@@H](O2)N3C=NC4=C(N=CN=C43)N)O)OP(=O)(O1)O" --output_dir $(pwd)/examples/ligand_binder/HalluDesign_op --num_seqs 2 --num_recycles 10 --ref_time_steps 50

# Antibody design targetting ptm peptide example
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH && python ./HalluDesign_run.py --HalluDesign_model af3 --input_file examples/antibody/ab_pser8_antibody.pdb --template_path examples/antibody/antibody_p8s_two_domains.json --mpnn "ligand_mpnn" --output_dir $(pwd)/examples/antibody/HalluDesign_design --cdr "A26-32 A52-57 A99-105 B25-32 B50-56 B90-95" --ptm C 6 P --fix_chain_index "C" --num_seqs 2 --num_recycles 15 --design_epoch_begin 10 --ref_time_steps 150

# Nanobody optimization example
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH && python ./HalluDesign_run.py --HalluDesign_model af3 --input_file examples/antibody/7G8.pdb --template_path examples/antibody/EGFR_nanobody.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/antibody/HalluDesign_design_op  --num_seqs 2 --num_recycles 15 --design_epoch_begin 10 --ref_time_steps 150 --fix_chain_index B --mpnn protein_mpnn --framework "EVQLVESGGGLVQAGGSLRLSCAAS" "AMGWFRQAPGKEREFVVAIN" "YYADSVKGRFTISRDNAKNTMYLQMNSLKPEDTAVYYCAAG" "DYWGQGTQVTVSS" --replace_MSA

# Cyclic peptide design example
# will use cyclic positional encoding and we will only the length of Chain A as input and random initialize the sequence
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH && python ./HalluDesign_run.py --HalluDesign_model af3 --input_file examples/cyclic_peptide/PDL1_cyclic_peptide.pdb --template_path examples/protein_binder/pdl1_protein.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/cyclic_peptide/HalluDesign_design --fix_chain_index "B" --num_seqs 2 --num_recycles 30 --design_epoch_begin 20 --ref_time_steps 150 --cyclic 1 --random_init

# Non head-tail cyclic peptide design example
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH && python ./HalluDesign_run.py --HalluDesign_model af3 --input_file examples/cyclic_peptide/MCL1_pep.pdb --template_path examples/cyclic_peptide/non_head_tail_cyclic_peptide.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/cyclic_peptide/HalluDesign_design --fix_chain_index "B" --num_seqs 2 --num_recycles 30 --design_epoch_begin 20 --ref_time_steps 150 --random_init --fix_res_index "A8" --ccd AZOR

# CoDP empowered ligand binder design with Protenix model example
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && export CC=$(which gcc) && export CXX=$(which g++) && python ./HalluDesign_run.py --HalluDesign_model protenix --input_file examples/ligand_binder/protein_ligand.pdb --template_path examples/ligand_binder/template_sm_protenix.json --mpnn "ligand_mpnn" --sm "C1[C@@H]2[C@H]([C@H]([C@@H](O2)N3C=NC4=C(N=CN=C43)N)O)OP(=O)(O1)O" --output_dir $(pwd)/examples/ligand_binder/HalluDesign_op --num_seqs 2 --num_recycles 10 --ref_time_steps 50 --CoDP

# CoDP empowered monomer design with AF3 model example
export XLA_PYTHON_CLIENT_PREALLOCATE=false && export XLA_CLIENT_MEM_FRACTION=0.8 && export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH && python ./HalluDesign_run.py --HalluDesign_model af3 --input_file examples/monomer/monomer.pdb --template_path examples/monomer/template_und.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/monomer/HalluDesign_op --num_seqs 2 --num_recycles 10 --ref_time_steps 50 --CoDP

# Symmetric protein design example
export CC=$(which gcc) && export CXX=$(which g++) && python ./HalluDesign_run.py --HalluDesign_model cross_model --input_file examples/Symmetric_protein/trimer.pdb --template_path examples/Symmetric_protein/template_und_protenix_3.json  --extra_json_path examples/Symmetric_protein/template_und_3.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/Symmetric_protein/HalluDesign_design --num_seqs 2 --num_recycles 10 --ref_time_steps 150 --design_epoch_begin 8 --symmetry_chains "A,B,C" --random_init

# Symmetric protein optimization example
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + 'nvidia/cuda_nvcc/bin')"):$PATH && python ./HalluDesign_run.py --HalluDesign_model af3 --input_file examples/Symmetric_protein/trimer.pdb --template_path  examples/Symmetric_protein/template_und_3.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/Symmetric_protein/HalluDesign_design --num_seqs 2 --num_recycles 10 --ref_time_steps 150 --design_epoch_begin 8 --symmetry_chains "A,B,C"

# Nanobody CDR design example
export CC=$(which gcc) && export CXX=$(which g++) && python ./HalluDesign_run.py --HalluDesign_model protenix --input_file examples/protein_binder/FGFR_nanobody.pdb --template_path examples/protein_binder/template_und_protenix_nanobody.json  --mpnn "protein_mpnn" --output_dir $(pwd)/examples/protein_binder/HalluDesign_op --num_seqs 2 --num_recycles 10 --ref_time_steps 150 --design_epoch_begin 7 --random_init --framework "QVQLVESGGGLVQPGGSLRLSCAAS" "SLGWFRQAPGQGLEAVAAIASMGGLTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAA" "WGQGTLVTVSS" --fix_chain_index "B"

# Repeat protein design example
export CC=$(which gcc) && export CXX=$(which g++) && python ./HalluDesign_run.py --HalluDesign_model cross_model --input_file examples/monomer/monomer.pdb --template_path examples/monomer/template_und_protenix.json  --mpnn "protein_mpnn" --output_dir $(pwd)/examples/monomer/HalluDesign_op --num_seqs 2 --num_recycles 10 --ref_time_steps 150 --random_init --symmetry_segments 4

# enzyme design epoch1 example, you should replace enzyem_protenix_batch_1.json sm_molecular with your own path
export CC=$(which gcc) && export CXX=$(which g++) && python ./HalluDesign_run.py --HalluDesign_model cross_model --input_file examples/enzyme_design/peptide_l160_1_hpos_39_43_114.pdb --template_path examples/enzyme_design/enzyem_protenix_batch_1.json  --extra_json_path examples/enzyme_design/enzyem_af3_batch_1.json  --mpnn "ligand_mpnn" --output_dir $(pwd)/examples/enzyme_design/HalluDesign_op --num_seqs 2 --num_recycles 10 --ref_time_steps 150 --design_epoch_begin 12 --random_init --enzyme_design

# enzyme design epoch2 example 
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH && python ./HalluDesign_run.py --HalluDesign_model af3 --input_file examples/enzyme_design/peptide_l160_1_hpos_39_43_114.pdb --template_path examples/enzyme_design/enzyem_af3_batch_1.json --sm "[Zn+2]" "CCCC(O)(OC1C=CC2C(C)=CC(OC=2C=1)=O)O" --template_for_eval examples/enzyme_design/enzyem_af3_batch_2.json --mpnn "ligand_mpnn" --output_dir $(pwd)/examples/enzyme_design/HalluDesign_op --num_seqs 2 --num_recycles 10 --ref_time_steps 150 --design_epoch_begin 7  --enzyme_design

# optimization binding affinity example, please check more in the data_preprocess.ipynb for how to prepare the bias summary csv file and use --fix_seq_file to incorporate the binding affinity information into design
export PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')"):$PATH && python ./HalluDesign_run.py --HalluDesign_model af3 --input_file examples/ligand_binder/copys/protein_ligand_1.pdb --template_path examples/ligand_binder/template_CCD.json --mpnn "ligand_mpnn" --ccd "CMP" --output_dir $(pwd)/examples/ligand_binder/HalluDesign_op_ccd --num_seqs 2 --num_recycles 4 --ref_time_steps 50 --fix_seq_file examples/ligand_binder/bias_summary_1_2_re.csv

