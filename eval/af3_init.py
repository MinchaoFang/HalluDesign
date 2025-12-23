import os
import sys
print(sys.path)
import argparse
import pickle
# Assuming af3_model is in your Python path or the same directory
from af3_model import AF3DesignerPack

def process_json_folder(input_json_folder, output_base_dir, dump_result,
                        ref_pdb_path=None, ref_time_steps=0, cyclic=0, num_samples=8):
    """
    Processes all JSON files in a given input folder using AF3DesignerPack.

    Args:
        input_json_folder (str): Path to the folder containing JSON files.
        output_base_dir (str): Base directory where individual output folders will be created.
        jax_compilation_dir (str): Directory for JAX compilation cache.
        ref_pdb_path (str, optional): Path to a reference PDB file. Defaults to None.
        ref_time_steps (int, optional): Reference time steps. Defaults to 0.
        cyclic (int, optional): Cyclic parameter. Defaults to 1.
        num_samples (int, optional): Number of samples to generate. Defaults to 8.
    """
    # Initialize the AF3DesignerPack model once
    jax_compilation_dir = "~/tmp"
    print(f"Initializing AF3DesignerPack with JAX compilation directory: {jax_compilation_dir}")
    model_af3 = AF3DesignerPack(jax_compilation_dir=jax_compilation_dir)
    print("AF3DesignerPack initialized.")

    # Iterate through all files in the input JSON folder
    for filename in input_json_folder:
        if filename.endswith(".json"):
            json_path = filename
            
            # Create a unique output directory for each JSON file
            # The output directory name will be based on the JSON filename (without extension)
            base_filename = os.path.splitext(filename)[0]
            current_out_dir = os.path.join(output_base_dir, base_filename)
            
            # Ensure the output directory exists
            os.makedirs(current_out_dir, exist_ok=True)
            
            print(f"\n--- Processing {filename} ---")
            print(f"  Input JSON: {json_path}")
            print(f"  Output Directory: {current_out_dir}")
            

            try:
                # Call the single_file_process method
                model_inference = model_af3.single_file_process(
                    json_path=json_path,
                    out_dir=current_out_dir,
                    ref_pdb_path=ref_pdb_path,
                    ref_time_steps=ref_time_steps,
                    cyclic=cyclic,
                    num_samples=num_samples
                )
                if dump_result:
                    with open(dump_result, "wb") as f:
                        pickle.dump(model_inference, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Successfully processed {filename}.")
                # You might want to do something with model_inference here if needed
                # For example, log results, collect data, etc.
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                print(f"Skipping {filename} and moving to the next.")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process multiple JSON files using AF3DesignerPack."
    )
    
    parser.add_argument(
        "--input_json_folder",
        type=str,
        help="Path to the folder containing JSON files to process."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        help="Path to the folder containing JSON files to process."
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        required=True,
        help="Base directory where individual output folders for each JSON will be created."
    )
    parser.add_argument(
        "--dump_result",
        type=str,
        required=True,
        help="result_op file path to dump the result."
    )
    parser.add_argument(
        "--ref_pdb_path",
        type=str,
        default=None,
        help="Path to a reference PDB file. Defaults to None."
    )

    parser.add_argument(
        "--ref_time_steps",
        type=int,
        default=0,
        help="Reference time steps. Defaults to 0."
    )
    
    parser.add_argument(
        "--cyclic",
        type=int,
        default=0,
        help="Cyclic parameter. Defaults to 1."
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Number of samples to generate for each JSON. Defaults to 8."
    )

    args = parser.parse_args()
    args.input_json_folder = [args.input_json]
    # Call the main processing function with arguments from argparse
    process_json_folder(
        input_json_folder=args.input_json_folder,
        output_base_dir=args.output_base_dir,
        dump_result=args.dump_result,
        ref_pdb_path=args.ref_pdb_path,
        ref_time_steps=args.ref_time_steps,
        cyclic=args.cyclic,
        num_samples=args.num_samples
    )