
from collections.abc import Callable, Iterable, Sequence
import csv
import dataclasses
import functools
import multiprocessing
import os
import pathlib
import shutil
import string
import textwrap
import time
import typing
from typing import Final, Protocol, Self, TypeVar, overload
from absl import app
from absl import flags
import sys

import os
current_folder = os.path.dirname(os.path.abspath(__file__))
src_folder = os.path.join(current_folder, 'src')
sys.path.insert(0, src_folder)

import numpy as np
import jax.numpy as jnp
from typing import Dict, Tuple, List, Optional
print(sys.path)
from alphafold3.common import base_config
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.constants import chemical_components
import alphafold3.cpp
from alphafold3.data import featurisation
from alphafold3.data import pipeline
from alphafold3.jax.attention import attention
from alphafold3.model import features
from alphafold3.model import params
from alphafold3.model import post_processing
from alphafold3.model.components import base_model
from alphafold3.model.components import utils
from alphafold3.model.diffusion import model as diffusion_model
print(alphafold3.cpp.__file__)
import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
#jax.config.update("jax_disable_jit", True)
import numpy as np
from typing import Dict, Any, Optional
import pickle


class StructureParser:
    def __init__(self):
        self.ATOM_RECORD_FORMAT = {
            'record_name': (0, 6),
            'atom_number': (6, 11),
            'atom_name': (12, 16),
            'alt_loc': (16, 17),
            'res_name': (17, 20),
            'chain_id': (21, 22),
            'res_number': (22, 26),
            'x': (30, 38),
            'y': (38, 46),
            'z': (46, 54),
            'occupancy': (54, 60),
            'temp_factor': (60, 66),
            'element': (76, 78)
        }
    
    def parse_pdb(self, file_path: str) -> Dict[str, List]:

        coords=0
                        
        return coords
    
    def parse_cif(self, file_path: str) -> Dict[str, List]:

        coords = 0
        
        return coords
    
           

    def process_file(self, file_path: str) -> Dict:
        if file_path.endswith('.pdb'):
            coords = self.parse_pdb(file_path)
        elif file_path.endswith('.cif'):
            coords = self.parse_cif(file_path)
        elif file_path.endswith('.pkl'):
            with open(file_path, "rb") as f:
                coords = pickle.load(f)
        else:
            raise ValueError("file must be .pdb or .cif or .pkl")
            
        return jnp.array(coords)



class StructureFeatureIntegrator:
    @staticmethod
    def process_ref_file(ref_pdb_path: str) -> Dict[str, Any]:

        parser = StructureParser()
        coords = parser.process_file(ref_pdb_path)
        ref_features = {'ref_atom_positions': coords
        }
        return ref_features
    
    @staticmethod
    def integrate_reference_features(
        featurised_example: Dict[str, Any],
        ref_structure_data: Dict[str, Any]
    ) -> Dict[str, Any]:

        integrated_features = dict(featurised_example)
        
        for key, value in ref_structure_data.items():
            integrated_features[key] = value
            
        
        return integrated_features
    
    
    @staticmethod
    def process_and_integrate(
        featurised_example: Dict[str, Any],
        ref_pdb_path: str,
        device: Optional[jax.Device] = None
    ) -> Dict[str, Any]:

        device = device or jax.devices()[0]
        ref_structure_data = StructureFeatureIntegrator.process_ref_file(ref_pdb_path)
        
        integrated_features = StructureFeatureIntegrator.integrate_reference_features(
            featurised_example,
            ref_structure_data
        )

        device_features = jax.device_put(
            jax.tree_util.tree_map(
                jnp.asarray,
                integrated_features
            ),
            device
        )
        return device_features
class ConfigurableModel(Protocol):
  """A model with a nested config class."""

  class Config(base_config.BaseConfig):
    ...

  def __call__(self, config: Config) -> Self:
    ...

  @classmethod
  def get_inference_result(
      cls: Self,
      batch: features.BatchDict,
      result: base_model.ModelResult,
      target_name: str = '',
  ) -> Iterable[base_model.InferenceResult]:
    ...


ModelT = TypeVar('ModelT', bound=ConfigurableModel)


def make_model_config(
    *,
    model_class: type[ModelT] = diffusion_model.Diffuser,
    flash_attention_implementation: attention.Implementation = 'triton',
    ref_time_steps= 0,
    num_samples= 5,
    cyclic = False,
    ref_time_evaluation = False
):
  config = model_class.Config()

  if hasattr(config, 'global_config'):
    config.global_config.flash_attention_implementation = (
        flash_attention_implementation
    )
  config.global_config.cyclic = False
  if cyclic:
    config.global_config.cyclic = cyclic
  config.heads.diffusion.eval.num_samples = num_samples
  if ref_time_steps !=0:
    config.heads.diffusion.eval.ref_time_steps = ref_time_steps
  if ref_time_evaluation:
    config.heads.diffusion.eval.ref_time_evaluation = ref_time_evaluation
    config.heads.diffusion.eval.num_samples = num_samples + 2
  return config


class ModelRunner:
  """Helper class to run structure prediction stages."""

  def __init__(
      self,
      model_class: ConfigurableModel,
      config: base_config.BaseConfig,
      device: jax.Device,
      model_dir: pathlib.Path,
  ):
    self._model_class = model_class
    self._model_config = config
    self._device = device
    self._model_dir = model_dir

  @functools.cached_property
  def model_params(self) -> hk.Params:
    """Loads model parameters from the model directory."""
    return params.get_model_haiku_params(model_dir=self._model_dir)

  @functools.cached_property
  def _model(
      self,
  ) -> Callable[[jnp.ndarray, features.BatchDict], base_model.ModelResult]:
    """Loads model parameters and returns a jitted model forward pass."""
    assert isinstance(self._model_config, self._model_class.Config)

    @hk.transform
    def forward_fn(batch):
      result = self._model_class(self._model_config)(batch)
      result['__identifier__'] = self.model_params['__meta__']['__identifier__']
      return result

    return functools.partial(
        jax.jit(forward_fn.apply, device=self._device), self.model_params
    )

  def run_inference(
    self, 
    featurised_example: features.BatchDict, 
    rng_key: jnp.ndarray, 
    ref_pdb_path: os.PathLike[str] | None,  # 
    ref_pkl_dump_path: os.PathLike[str] | None,
) -> base_model.ModelResult:
    """Computes a forward pass of the model on a featurised example."""
    featurised_example = jax.device_put(
        jax.tree_util.tree_map(
            jnp.asarray, utils.remove_invalidly_typed_feats(featurised_example)
        ),
        self._device,
    )

    if ref_pdb_path :
      print(f'Running ref_pdb guided prediction '+ref_pdb_path)
      if not os.path.exists(ref_pdb_path):
        raise FileNotFoundError(f"Ref PDB file not found: {ref_pdb_path}")
      integrator = StructureFeatureIntegrator()
      final_features = integrator.process_and_integrate(
    featurised_example,
    ref_pdb_path,
      )
      result = self._model(rng_key, final_features)
    else:
      print(f'No ref_pdb guided prediction')

      result = self._model(rng_key, featurised_example)
    result = jax.tree.map(np.asarray, result)
    result = jax.tree.map(
        lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x,
        result,
    )
    result['__identifier__'] = result['__identifier__'].tobytes()
    print(result["diffusion_samples"]["atom_positions"].shape)
    if ref_pkl_dump_path:
        atom_positions = np.array(
            jax.device_get(result["diffusion_samples"]["atom_positions"]),
            dtype=np.float32  # 
        )
        print(f"dump pkl file in {ref_pkl_dump_path}")
        with open(ref_pkl_dump_path, 'wb') as f:
            pickle.dump(atom_positions, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result

  def extract_structures(
      self,
      batch: features.BatchDict,
      result: base_model.ModelResult,
      target_name: str,
  ) -> list[base_model.InferenceResult]:
    """Generates structures from model outputs."""
    return list(
        self._model_class.get_inference_result(
            batch=batch, result=result, target_name=target_name
        )
    )
  def update_config(self,config: base_config.BaseConfig):
    self._model_config = config
    


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
  """Stores the inference results (diffusion samples) for a single seed.

  Attributes:
    seed: The seed used to generate the samples.
    inference_results: The inference results, one per sample.
    full_fold_input: The fold input that must also include the results of
      running the data pipeline - MSA and templates.
  """

  seed: int
  inference_results: Sequence[base_model.InferenceResult]
  full_fold_input: folding_input.Input


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    ref_pdb_path: os.PathLike[str] | str,
    ref_pkl_dump_path: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
  """Runs the full inference pipeline to predict structures for each seed."""

  print(f'Featurising data for seeds {fold_input.rng_seeds}...')
  featurisation_start_time = time.time()
  ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
  featurised_examples = featurisation.featurise_input(
      fold_input=fold_input, buckets=buckets, ccd=ccd, verbose=True
  )
  print(
      f'Featurising data for seeds {fold_input.rng_seeds} took '
      f' {time.time() - featurisation_start_time:.2f} seconds.'
  )
  all_inference_start_time = time.time()
  all_inference_results = []
  for seed, example in zip(fold_input.rng_seeds, featurised_examples):
    print(f'Running model inference for seed {seed}...')
    inference_start_time = time.time()
    rng_key = jax.random.PRNGKey(seed)
    result = model_runner.run_inference(example, rng_key,ref_pdb_path,ref_pkl_dump_path)
    print(
        f'Running model inference for seed {seed} took '
        f' {time.time() - inference_start_time:.2f} seconds.'
    )
    print(f'Extracting output structures (one per sample) for seed {seed}...')
    extract_structures = time.time()
    inference_results = model_runner.extract_structures(
        batch=example, result=result, target_name=fold_input.name
    )
    print(
        f'Extracting output structures (one per sample) for seed {seed} took '
        f' {time.time() - extract_structures:.2f} seconds.'
    )
    all_inference_results.append(
        ResultsForSeed(
            seed=seed,
            inference_results=inference_results,
            full_fold_input=fold_input,
        )
    )
    print(
        'Running model inference and extracting output structures for seed'
        f' {seed} took  {time.time() - inference_start_time:.2f} seconds.'
    )
  print(
      'Running model inference and extracting output structures for seeds'
      f' {fold_input.rng_seeds} took '
      f' {time.time() - all_inference_start_time:.2f} seconds.'
  )
  return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
  """Writes the input JSON to the output directory."""
  os.makedirs(output_dir, exist_ok=True)
  with open(
      os.path.join(output_dir, f'{fold_input.sanitised_name()}_data.json'), 'wt'
  ) as f:
    f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
  """Writes outputs to the specified output directory."""
  ranking_scores = []
  max_ranking_score = None
  max_ranking_result = None

  output_terms = (
      pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
  ).read_text()

  os.makedirs(output_dir, exist_ok=True)
  for results_for_seed in all_inference_results:
    seed = results_for_seed.seed
    for sample_idx, result in enumerate(results_for_seed.inference_results):
      sample_dir = os.path.join(output_dir, f'all_samples')
      
      os.makedirs(sample_dir, exist_ok=True)
      post_processing.write_output(
          inference_result=result, output_dir=sample_dir,name=f'seed-{seed}_sample-{sample_idx}'
      )
      ranking_score = float(result.metadata['ranking_score'])
      ranking_scores.append((seed, sample_idx, ranking_score))
      if max_ranking_score is None or ranking_score > max_ranking_score:
        max_ranking_score = ranking_score
        max_ranking_result = result

  if max_ranking_result is not None:  # True iff ranking_scores non-empty.
    post_processing.write_output(
        inference_result=max_ranking_result,
        output_dir=output_dir,
        # The output terms of use are the same for all seeds/samples.         terms_of_use=output_terms,
        name=job_name,
    )
    # Save csv of ranking scores with seeds and sample indices, to allow easier
    # comparison of ranking scores across different runs.
    #with open(os.path.join(output_dir, 'ranking_scores.csv'), 'wt') as f:
    #  writer = csv.writer(f)
    #  writer.writerow(['seed', 'sample', 'ranking_score'])
    #  writer.writerows(ranking_scores)


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> folding_input.Input:
  ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
) -> Sequence[ResultsForSeed]:
  ...


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    ref_pdb_path: os.PathLike[str] | None,
    ref_pkl_dump_path: os.PathLike[str] | None,
    buckets: Sequence[int] | None = None,
) -> folding_input.Input | Sequence[ResultsForSeed]:
  """Runs data pipeline and/or inference on a single fold input.

  Args:
    fold_input: Fold input to process.
    data_pipeline_config: Data pipeline config to use. If None, skip the data
      pipeline.
    model_runner: Model runner to use. If None, skip inference.
    output_dir: Output directory to write to.
    buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
      of the model. If None, calculate the appropriate bucket size from the
      number of tokens. If not None, must be a sequence of at least one integer,
      in strictly increasing order. Will raise an error if the number of tokens
      is more than the largest bucket size.

  Returns:
    The processed fold input, or the inference results for each seed.

  Raises:
    ValueError: If the fold input has no chains.
  """
  print(f'Processing fold input {fold_input.name}')

  if not fold_input.chains:
    raise ValueError('Fold input has no chains.')

  if model_runner is not None:
    # If we're running inference, check we can load the model parameters before
    # (possibly) launching the data pipeline.
    print('Checking we can load the model parameters...')
    start_time = time.time()
    _ = model_runner.model_params

    first_load_time = time.time() - start_time

    print(f'✅ loading checkpoint of af3: {first_load_time:.4f} s')
    

  if data_pipeline_config is None:
    print('Skipping data pipeline...')
  else:
    print('Running data pipeline...')
    fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

  print(f'Output directory: {output_dir}')
  #print(f'Writing model input JSON to {output_dir}')
  #write_fold_input_json(fold_input, output_dir)
  if model_runner is None:
    print('Skipping inference...')
    output = fold_input
  else:
    print(
        f'Predicting 3D structure for {fold_input.name} for seed(s)'
        f' {fold_input.rng_seeds}...'
    )
    all_inference_results = predict_structure(
        fold_input=fold_input,
        model_runner=model_runner,
        ref_pdb_path=ref_pdb_path,
        ref_pkl_dump_path=ref_pkl_dump_path,
        buckets=buckets)
    print(
        f'Writing outputs for {fold_input.name} for seed(s)'
        f' {fold_input.rng_seeds}...'
    )
    write_outputs(
        all_inference_results=all_inference_results,
        output_dir=output_dir,
        job_name=fold_input.sanitised_name(),
    )
    output = all_inference_results

  print(f'Done processing fold input {fold_input.name}.')
  return output


    
class AF3DesignerPack:
    def __init__(self,jax_compilation_dir):
        self.devices = jax.local_devices(backend='gpu')
        home_dir = os.path.expanduser("~")
        self._HOME_DIR = pathlib.Path(home_dir)
        self._MODEL_DIR =  self._HOME_DIR / 'model'


        self._FLASH_ATTENTION_IMPLEMENTATION = 'triton'
        jax.config.update(
        'jax_compilation_cache_dir', jax_compilation_dir
        )
        self._BUCKETS: Final[tuple[int, ...]] = (
            256,
            512,
            768,
            1024,
            1280,
            1536,
            2048,
            2560,
            3072,
            3584,
            4096,
            4608,
            5120,
        )
        print(f'Found local devices: {[d.device_kind for d in self.devices]}')  # 
        config = make_model_config(
                flash_attention_implementation=typing.cast(
                    attention.Implementation, self._FLASH_ATTENTION_IMPLEMENTATION
                )
            )
        self.model_runner = ModelRunner(
            model_class=diffusion_model.Diffuser,
            config=config,
            device=self.devices[0],
            model_dir=pathlib.Path(self._MODEL_DIR),
        )

        start_time = time.time()
        _= self.model_runner.model_params

        first_load_time = time.time() - start_time

        print(f'✅ loading checkpoint of af3: {first_load_time:.4f} s')


        
    def single_file_process(self,json_path,out_dir,
                            ref_pdb_path=None,
                            ref_time_steps =0,
                            num_samples=5,
                            ref_time_evaluation = 0,
                            cyclic = None,
                            ref_pkl_dump_path=None):
        """"""

        fold_inputs = folding_input.load_fold_inputs_from_path(
            pathlib.Path(json_path)
        )
        
        config_update = make_model_config(
                flash_attention_implementation=typing.cast(
                    attention.Implementation, self._FLASH_ATTENTION_IMPLEMENTATION),
                ref_time_steps=ref_time_steps,
                num_samples=num_samples,
                cyclic=cyclic,
                ref_time_evaluation=ref_time_evaluation

            )
        self.model_runner.update_config(config_update)
        
        inference_output = None

        for fold_input in fold_inputs:
            inference_output = process_fold_input(
                fold_input=fold_input,
                data_pipeline_config=None,  
                model_runner=self.model_runner,
                output_dir=os.path.join(out_dir, fold_input.sanitised_name()),
                ref_pdb_path=ref_pdb_path if ref_pdb_path else None,
                ref_pkl_dump_path=ref_pkl_dump_path if ref_pkl_dump_path else None,
                buckets=self._BUCKETS if self._BUCKETS else []
            )
        return inference_output


  

