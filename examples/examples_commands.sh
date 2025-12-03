# Protein Monomer optmization exmample
python /storage/caolab/fangmc/for_pub/HalluDesign/HalluDesign_run.py --prediction_model af3 --input_file examples/monomer/monomer.pdb --template_path examples/monomer/template_und.json --output_dir /storage/caolab/fangmc/tmp/test --num_seqs 2 --num_recycles 10 --ref_time_steps 50

# Protein binder optmization exmample
python /storage/caolab/fangmc/for_pub/HalluDesign/HalluDesign_run.py --prediction_model af3 --input_file examples/protein_binder/Pdl1_binder_design_5.pdb --template_path examples/protein_binder/pdl1_protein.json --output_dir examples/protein_binder/HalluDesign_op --fix_chain_index "B" --num_seqs 2 --num_recycles 10 --ref_time_steps 50 

# Ligand binder optimization example with CCD ligand

# Ligand binder optimization example with smiles ligand