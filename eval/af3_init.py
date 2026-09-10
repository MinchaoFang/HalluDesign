import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import pickle
import json
# Assuming af3_model is in your Python path or the same directory
from af3_model import AF3DesignerPack
from motif_constraints import inject_af3_motif_template, load_motif_spec

def process_json_folder(input_json_folder, output_base_dir, dump_result,
                        ref_pdb_path=None, ref_time_steps=0, cyclic=0, num_samples=8,
                        motif_spec_path="", current_structure_path=None,
                        soft_projection_weight=None):
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
    motif_spec = load_motif_spec(motif_spec_path) if motif_spec_path else None
    if motif_spec is not None and soft_projection_weight is not None:
        motif_spec = motif_spec.with_runtime_options(
            soft_projection_weight=soft_projection_weight
        )

    # Iterate through all files in the input JSON folder
    for filename in input_json_folder:
        if filename.endswith(".json"):
            json_path = filename
            
            # Create a unique output directory for each JSON file
            # The output directory name will be based on the JSON filename (without extension)
            base_filename = os.path.splitext(os.path.basename(filename))[0]
            current_out_dir = os.path.join(output_base_dir, base_filename)
            
            # Ensure the output directory exists
            os.makedirs(current_out_dir, exist_ok=True)
            
            print(f"\n--- Processing {filename} ---")
            print(f"  Input JSON: {json_path}")
            print(f"  Output Directory: {current_out_dir}")
            

            try:
                inference_json_path = json_path
                if motif_spec is not None:
                    # Keep the caller's JSON unchanged while preparing the
                    # AF3 input used by the cross-model subprocess.
                    with open(json_path, "r", encoding="utf-8") as handle:
                        input_json = json.load(handle)
                    inject_af3_motif_template(
                        input_json, motif_spec, motif_spec.uses("template")
                    )
                    inference_json_path = os.path.join(
                        current_out_dir, f"{base_filename}_motif_input.json"
                    )
                    with open(inference_json_path, "w", encoding="utf-8") as handle:
                        json.dump(input_json, handle, indent=2)

                # Call the single_file_process method
                model_inference = model_af3.single_file_process(
                    json_path=inference_json_path,
                    out_dir=current_out_dir,
                    ref_pdb_path=ref_pdb_path,
                    ref_time_steps=ref_time_steps,
                    cyclic=cyclic,
                    num_samples=num_samples,
                    motif_spec=(
                        motif_spec if motif_spec is not None
                        and (
                            motif_spec.uses("af3_projection")
                            or motif_spec.uses("af3_soft_projection")
                        ) else None
                    ),
                    current_structure_path=current_structure_path,
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
    parser.add_argument("--motif_spec", type=str, default="",
                        help="Optional motif JSON used for AF3 motif constraints.")
    parser.add_argument(
        "--current_structure_path",
        type=str,
        default=None,
        help="Current scaffold PDB/CIF used to align motif coordinates.",
    )
    parser.add_argument(
        "--soft_projection_weight",
        type=float,
        default=None,
        help="Optional runtime override for AF3 soft projection.",
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
        num_samples=args.num_samples,
        motif_spec_path=args.motif_spec,
        current_structure_path=args.current_structure_path,
        soft_projection_weight=args.soft_projection_weight,
    )
