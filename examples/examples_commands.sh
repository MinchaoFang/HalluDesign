# Protein Monomer optmization exmample
python /storage/caolab/fangmc/for_pub/HalluDesign/HalluDesign_run.py --prediction_model af3 --input_file examples/monomer/monomer.pdb --template_path examples/monomer/template_und.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/monomer/HalluDesign_op --num_seqs 2 --num_recycles 10 --ref_time_steps 50

# Protein binder optmization exmample
python /storage/caolab/fangmc/for_pub/HalluDesign/HalluDesign_run.py --prediction_model af3 --input_file examples/protein_binder/Pdl1_binder_design_5.pdb --template_path examples/protein_binder/pdl1_protein.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/protein_binder/HalluDesign_op --fix_chain_index "B" --num_seqs 2 --num_recycles 10 --ref_time_steps 50 

# Ligand binder optimization example with CCD ligand
python /storage/caolab/fangmc/for_pub/HalluDesign/HalluDesign_run.py --prediction_model af3 --input_file examples/ligand_binder/protein_ligand.pdb --template_path examples/ligand_binder/template_CCD.json --mpnn "ligand_mpnn" --ccd "CMP" --output_dir $(pwd)/examples/ligand_binder/HalluDesign_op_ccd --num_seqs 2 --num_recycles 10 --ref_time_steps 50

# Ligand binder optimization example with smiles ligand
python /storage/caolab/fangmc/for_pub/HalluDesign/HalluDesign_run.py --prediction_model af3 --input_file examples/ligand_binder/protein_ligand.pdb --template_path examples/ligand_binder/template_sm.json --mpnn "ligand_mpnn" --sm "C1[C@@H]2[C@H]([C@H]([C@@H](O2)N3C=NC4=C(N=CN=C43)N)O)OP(=O)(O1)O"  --output_dir $(pwd)/examples/ligand_binder/HalluDesign_op_sm --num_seqs 2 --num_recycles 10 --ref_time_steps 50

# Antibody design example
python /storage/caolab/fangmc/for_pub/HalluDesign/HalluDesign_run.py --prediction_model af3 --input_file examples/antibody/ab_pser8_antibody.pdb --template_path examples/antibody/antibody_p8s_two_domains.json --mpnn "ligand_mpnn" --output_dir $(pwd)/examples/antibody/HalluDesign_design --cdr "A26-32 A52-57 A99-105 B25-32 B50-56 B90-95" --ptm C 6 P --fix_chain_index "C" --num_seqs 2 --num_recycles 15 --design_epoch_begin 10 --ref_time_steps 150

# Nanobody optimization example
python /storage/caolab/fangmc/for_pub/HalluDesign/HalluDesign_run.py --prediction_model af3 --input_file examples/antibody/7G8.pdb --template_path examples/antibody/EGFR_nanobody.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/antibody/HalluDesign_design_op  --num_seqs 2 --num_recycles 15 --design_epoch_begin 10 --ref_time_steps 150 --fix_chain_index B --mpnn protein_mpnn --framework "EVQLVESGGGLVQAGGSLRLSCAAS" "AMGWFRQAPGKEREFVVAIN" "YYADSVKGRFTISRDNAKNTMYLQMNSLKPEDTAVYYCAAG" "DYWGQGTQVTVSS" --replace_MSA

# Cyclic peptide design example
# will use cyclic positional encoding and we will only the length of Chain A as input
python /storage/caolab/fangmc/for_pub/HalluDesign/HalluDesign_run.py --prediction_model af3 --input_file examples/cyclic_peptide/PDL1_cyclic_peptide.pdb --template_path examples/protein_binder/pdl1_protein.json --mpnn "protein_mpnn" --output_dir $(pwd)/examples/cyclic_peptide/HalluDesign_design --fix_chain_index "B" --num_seqs 2 --num_recycles 30 --design_epoch_begin 20 --ref_time_steps 150 --cyclic 1 --random_init