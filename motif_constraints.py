"""Utilities for optional motif scaffolding and AF3 template constraints.

The normal HalluDesign path does not use this module. Motifs are converted to
single-chain AF3 mmCIF templates and mapped by chain/residue identifiers.
"""

from __future__ import annotations

import json
import os
import copy
import string
from dataclasses import dataclass, replace
from io import StringIO
from typing import Any, Mapping

import numpy as np
from Bio.PDB import Chain, MMCIFIO, MMCIFParser, Model, PDBParser, Structure


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "MSE": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

_CONSTRAINT_MODES = frozenset({"template", "af3_projection", "protenix_projection"})
_MOTIF_MODES = frozenset({"off", "refine", "scaffold_then_refine"})


def _parse_atom_key(value: str) -> tuple[str, int, str]:
    """Parse ``chain:residue[:atom]`` identifiers used by motif specs."""
    parts = str(value).replace("/", ":").split(":")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"Invalid atom/residue identifier {value!r}; expected chain:residue[:atom]."
        )
    chain = parts[0].strip()
    residue = parts[1].strip()
    if not chain or not residue:
        raise ValueError(f"Invalid atom/residue identifier: {value!r}")
    try:
        res_id = int(residue)
    except ValueError as exc:
        raise ValueError(f"Residue id must be an integer in {value!r}.") from exc
    atom = parts[2].strip() if len(parts) == 3 else ""
    return chain, res_id, atom


def _load_structure(path: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Motif structure not found: {path}")
    parser = MMCIFParser(QUIET=True) if path.lower().endswith(".cif") else PDBParser(QUIET=True)
    return parser.get_structure("motif", path)


def _structure_atom_index(path: str) -> dict[tuple[str, int, str], np.ndarray]:
    structure = _load_structure(path)
    atom_index: dict[tuple[str, int, str], np.ndarray] = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = int(residue.id[1])
                for atom in residue:
                    key = (str(chain.id), res_id, str(atom.name).strip())
                    atom_index.setdefault(key, np.asarray(atom.coord, dtype=np.float32))
        # Only the first model is used, matching the rest of the repository.
        break
    return atom_index


@dataclass(frozen=True)
class MotifMapping:
    source: tuple[str, int, str]
    target: tuple[str, int, str]


@dataclass(frozen=True)
class MotifSpec:
    path: str
    motif_pdb: str
    mappings: tuple[MotifMapping, ...]
    fixed_atom_policy: str = "all"
    fixed_atom_names: tuple[str, ...] = ()
    constraint_modes: frozenset[str] = frozenset({"template"})
    motif_mode: str = "off"
    scaffold_cycles: int = 1
    scaffold_steps: int = 150
    refine_steps: int = 50

    @classmethod
    def from_file(cls, path: str) -> "MotifSpec":
        if not path:
            raise ValueError("A motif spec path is required when motif constraints are enabled.")
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, Mapping):
            raise ValueError("Motif spec must contain a JSON object.")

        motif_pdb = str(raw.get("motif_pdb", ""))
        if not motif_pdb:
            raise ValueError("Motif spec must contain motif_pdb.")
        if not os.path.isabs(motif_pdb):
            spec_relative = os.path.abspath(
                os.path.join(os.path.dirname(os.path.abspath(path)), motif_pdb)
            )
            motif_pdb = (
                spec_relative
                if os.path.exists(spec_relative)
                else os.path.abspath(motif_pdb)
            )
        if not os.path.exists(motif_pdb):
            raise FileNotFoundError(f"Motif structure not found: {motif_pdb}")

        raw_mappings = raw.get("residue_mapping", raw.get("mappings", []))
        if not raw_mappings:
            raise ValueError("Motif spec must contain a non-empty residue_mapping list.")
        mappings: list[MotifMapping] = []
        for item in raw_mappings:
            if not isinstance(item, Mapping) or "source" not in item or "target" not in item:
                raise ValueError("Each motif mapping must contain source and target.")
            source = _parse_atom_key(item["source"])
            target = _parse_atom_key(item["target"])
            if source[2] and target[2] and source[2] != target[2]:
                raise ValueError(
                    f"Source/target atom names differ in mapping {item!r}; use explicit atom mappings only when intended."
                )
            if source[2] and not target[2]:
                target = (target[0], target[1], source[2])
            elif target[2] and not source[2]:
                source = (source[0], source[1], target[2])
            mappings.append(MotifMapping(source=source, target=target))

        policy = str(raw.get("fixed_atom_policy", "all")).lower()
        if policy not in {"all", "backbone", "custom"}:
            raise ValueError("fixed_atom_policy must be one of: all, backbone, custom.")
        raw_fixed_atoms = raw.get("fixed_atoms", [])
        if isinstance(raw_fixed_atoms, str):
            raw_fixed_atoms = [raw_fixed_atoms]
        if not isinstance(raw_fixed_atoms, (list, tuple, set)):
            raise ValueError("fixed_atoms must be a list of atom names.")
        fixed_atom_names = tuple(
            str(name).strip().upper() for name in raw_fixed_atoms if str(name).strip()
        )
        if policy == "custom" and not fixed_atom_names:
            raise ValueError("fixed_atoms is required when fixed_atom_policy is custom.")
        raw_constraint_modes = raw.get("constraint_modes", ["template"])
        if raw_constraint_modes is None:
            raw_constraint_modes = []
        if isinstance(raw_constraint_modes, str):
            raw_constraint_modes = [raw_constraint_modes]
        if not isinstance(raw_constraint_modes, (list, tuple, set)):
            raise ValueError("constraint_modes must be a list of supported values.")
        constraint_modes = frozenset(str(mode).lower() for mode in raw_constraint_modes)
        unknown_modes = constraint_modes - _CONSTRAINT_MODES
        if unknown_modes:
            raise ValueError(
                "constraint_modes contains unsupported values: "
                f"{sorted(unknown_modes)}. Supported values: {sorted(_CONSTRAINT_MODES)}."
            )
        motif_mode = str(raw.get("motif_mode", "off")).lower()
        if motif_mode not in _MOTIF_MODES:
            raise ValueError(f"motif_mode must be one of: {sorted(_MOTIF_MODES)}.")
        scaffold_cycles = int(raw.get("scaffold_cycles", 1))
        scaffold_steps = int(raw.get("scaffold_steps", 150))
        refine_steps = int(raw.get("refine_steps", 50))
        if scaffold_cycles < 0:
            raise ValueError("scaffold_cycles must be non-negative.")
        if scaffold_steps <= 0 or refine_steps <= 0:
            raise ValueError("scaffold_steps and refine_steps must be positive.")
        target_atoms = [mapping.target for mapping in mappings]
        if len(target_atoms) != len(set(target_atoms)):
            raise ValueError("residue_mapping contains duplicate target atoms.")
        return cls(
            path=os.path.abspath(path),
            motif_pdb=motif_pdb,
            mappings=tuple(mappings),
            fixed_atom_policy=policy,
            fixed_atom_names=fixed_atom_names,
            constraint_modes=constraint_modes,
            motif_mode=motif_mode,
            scaffold_cycles=scaffold_cycles,
            scaffold_steps=scaffold_steps,
            refine_steps=refine_steps,
        )

    def with_runtime_options(
        self,
        motif_mode: str | None = None,
        scaffold_steps: int | None = None,
        refine_steps: int | None = None,
    ) -> "MotifSpec":
        """Apply optional command-line overrides without changing the JSON file."""
        updates = {}
        if motif_mode is not None:
            motif_mode = motif_mode.lower()
            if motif_mode not in _MOTIF_MODES:
                raise ValueError(f"motif_mode must be one of: {sorted(_MOTIF_MODES)}.")
            updates["motif_mode"] = motif_mode
        if scaffold_steps is not None:
            if scaffold_steps <= 0:
                raise ValueError("scaffold_steps must be positive.")
            updates["scaffold_steps"] = scaffold_steps
        if refine_steps is not None:
            if refine_steps <= 0:
                raise ValueError("refine_steps must be positive.")
            updates["refine_steps"] = refine_steps
        return replace(self, **updates)

    def uses(self, constraint_mode: str) -> bool:
        return constraint_mode.lower() in self.constraint_modes

    def phase_for_cycle(self, cycle: int) -> str:
        if self.motif_mode == "scaffold_then_refine" and cycle < self.scaffold_cycles:
            return "scaffold"
        if self.motif_mode in {"refine", "scaffold_then_refine"}:
            return "refine"
        return "legacy"

    def diffusion_steps_for_cycle(self, cycle: int, legacy_steps: int) -> int:
        phase = self.phase_for_cycle(cycle)
        if phase == "scaffold":
            return self.scaffold_steps
        if phase == "refine":
            return self.refine_steps
        return legacy_steps

    @property
    def source_residues(self) -> set[tuple[str, int]]:
        return {(m.source[0], m.source[1]) for m in self.mappings}

    @property
    def target_residues(self) -> set[tuple[str, int]]:
        return {(m.target[0], m.target[1]) for m in self.mappings}

    def source_coordinates(self) -> dict[tuple[str, int, str], np.ndarray]:
        source_atoms = _structure_atom_index(self.motif_pdb)
        result: dict[tuple[str, int, str], np.ndarray] = {}
        missing = []
        allowed_atoms = None
        if self.fixed_atom_policy == "backbone":
            allowed_atoms = {"N", "CA", "C", "O"}
        elif self.fixed_atom_policy == "custom":
            allowed_atoms = set(self.fixed_atom_names)
        for mapping in self.mappings:
            source_key = mapping.source
            if source_key[2]:
                coord = source_atoms.get(source_key)
                if coord is None:
                    missing.append(source_key)
                elif allowed_atoms is not None and source_key[2] not in allowed_atoms:
                    raise ValueError(
                        f"Mapped atom {source_key[2]!r} is excluded by "
                        f"fixed_atom_policy={self.fixed_atom_policy!r}."
                    )
                elif mapping.target in result:
                    raise ValueError(f"Duplicate target atom mapping: {mapping.target}")
                else:
                    result[mapping.target] = coord
                continue
            residue_atoms = {
                key[2]: coord
                for key, coord in source_atoms.items()
                if key[:2] == source_key[:2]
            }
            if not residue_atoms:
                missing.append(source_key)
                continue
            for atom_name, coord in residue_atoms.items():
                if allowed_atoms is None or atom_name in allowed_atoms:
                    target_key = (mapping.target[0], mapping.target[1], atom_name)
                    if target_key in result:
                        raise ValueError(f"Duplicate target atom mapping: {target_key}")
                    result[target_key] = coord
        if missing:
            raise ValueError(f"Motif atoms/residues were not found: {missing[:8]}")
        return result

    def target_residue_strings(self) -> list[str]:
        return [f"{chain}{res_id}" for chain, res_id in sorted(self.target_residues)]

    @property
    def target_chain(self) -> str:
        chains = {chain for chain, _ in self.target_residues}
        if len(chains) != 1:
            raise ValueError(
                "An AF3 motif template must target exactly one protein chain; "
                f"got {sorted(chains)}."
            )
        return next(iter(chains))

    def _template_structure(self):
        """Return the mapped motif as one chain with contiguous residue IDs."""
        source_chains = {chain for chain, _ in self.source_residues}
        if len(source_chains) != 1:
            raise ValueError(
                "An AF3 motif template must contain exactly one source chain; "
                f"got {sorted(source_chains)}."
            )
        source_chain_id = next(iter(source_chains))
        allowed_atoms = None
        if self.fixed_atom_policy == "backbone":
            allowed_atoms = {"N", "CA", "C", "O"}
        elif self.fixed_atom_policy == "custom":
            allowed_atoms = set(self.fixed_atom_names)

        source_structure = _load_structure(self.motif_pdb)
        source_model = next(source_structure.get_models())
        if source_chain_id not in source_model:
            raise ValueError(
                f"Motif source chain {source_chain_id!r} was not found in "
                f"{self.motif_pdb}."
            )

        wanted = self.source_residues
        selected_source_keys = []
        selected_residues = {}
        source_chain = source_model[source_chain_id]
        for residue in source_chain:
            residue_key = (source_chain_id, int(residue.id[1]))
            if residue_key not in wanted:
                continue
            selected_source_keys.append(residue_key)
            selected_residues[residue_key] = residue

        missing = sorted(wanted - set(selected_residues))
        if missing:
            raise ValueError(
                f"Motif residues were not found in {self.motif_pdb}: {missing}"
            )

        motif_structure = Structure.Structure("motif_template")
        motif_model = Model.Model(0)
        motif_chain = Chain.Chain("A")
        for template_index, source_key in enumerate(selected_source_keys, start=1):
            residue = copy.deepcopy(selected_residues[source_key])
            residue.id = (" ", template_index, " ")
            if allowed_atoms is not None:
                for atom in list(residue):
                    if atom.name.strip() not in allowed_atoms:
                        residue.detach_child(atom.id)
            if len(residue) == 0:
                raise ValueError(f"Motif residue has no retained atoms: {source_key}")
            motif_chain.add(residue)
        motif_model.add(motif_chain)
        motif_structure.add(motif_model)
        return motif_structure, selected_source_keys

    def build_af3_template(self, query_sequence_length: int) -> dict[str, Any]:
        """Build an AF3 JSON template from the mapped motif residues.

        AF3 expects template residue indices to refer to the residue order in a
        single-chain mmCIF, while query indices refer to the target sequence.
        The extracted motif is therefore renumbered continuously from one.
        """
        if query_sequence_length < 1:
            raise ValueError("The AF3 target sequence must not be empty.")
        self.source_coordinates()
        motif_structure, selected_source_keys = self._template_structure()
        output = StringIO()
        mmcif_io = MMCIFIO()
        mmcif_io.set_structure(motif_structure)
        mmcif_io.save(output)

        source_to_template = {
            source_key: index
            for index, source_key in enumerate(selected_source_keys)
        }
        query_indices = []
        template_indices = []
        seen_targets = set()
        target_to_source = {}
        for mapping in self.mappings:
            target_residue = (mapping.target[0], mapping.target[1])
            source_residue = (mapping.source[0], mapping.source[1])
            previous_source = target_to_source.get(target_residue)
            if previous_source is not None and previous_source != source_residue:
                raise ValueError(
                    f"Target residue {target_residue} maps to multiple source "
                    f"residues: {previous_source} and {source_residue}."
                )
            if target_residue in seen_targets:
                continue
            target_to_source[target_residue] = source_residue
            if target_residue[0] != self.target_chain:
                raise ValueError(
                    "All motif mappings must target the same AF3 protein chain."
                )
            query_index = target_residue[1] - 1
            if not 0 <= query_index < query_sequence_length:
                raise ValueError(
                    f"Target residue {target_residue} is outside the AF3 sequence "
                    f"of length {query_sequence_length}."
                )
            query_indices.append(query_index)
            template_indices.append(source_to_template[source_residue])
            seen_targets.add(target_residue)

        if not query_indices:
            raise ValueError("The motif template has no residue mappings.")
        return {
            "mmcif": output.getvalue(),
            "queryIndices": query_indices,
            "templateIndices": template_indices,
        }

    def source_residue_sequence(self) -> dict[tuple[str, int], str]:
        """Return one-letter amino acids for source residues used by the motif."""
        source_atoms = _load_structure(self.motif_pdb)
        source_model = next(source_atoms.get_models())
        result = {}
        for chain_id, res_id in self.source_residues:
            if chain_id not in source_model:
                raise ValueError(f"Motif source chain {chain_id!r} was not found.")
            residue = next(
                (candidate for candidate in source_model[chain_id]
                 if int(candidate.id[1]) == res_id),
                None,
            )
            if residue is None:
                raise ValueError(f"Motif residue {chain_id}:{res_id} was not found.")
            try:
                result[(chain_id, res_id)] = _THREE_TO_ONE[residue.resname.upper()]
            except KeyError as exc:
                raise ValueError(
                    f"Unsupported protein residue {residue.resname!r} at "
                    f"{chain_id}:{res_id}."
                ) from exc
        return result

def load_motif_spec(path: str | None) -> MotifSpec | None:
    return MotifSpec.from_file(path) if path else None


def build_af3_dense_projection(spec: MotifSpec, token_atoms_layout) -> tuple[np.ndarray, np.ndarray]:
    """Map motif atoms to AF3's actual dense atom layout."""
    coordinates = spec.source_coordinates()
    atom_names = np.asarray(token_atoms_layout.atom_name)
    chain_ids = np.asarray(token_atoms_layout.chain_id)
    residue_ids = np.asarray(token_atoms_layout.res_id)
    positions = np.zeros((*atom_names.shape, 3), dtype=np.float32)
    mask = np.zeros(atom_names.shape, dtype=bool)
    for index in np.ndindex(atom_names.shape):
        key = (str(chain_ids[index]), int(residue_ids[index]), str(atom_names[index]).strip())
        coordinate = coordinates.get(key)
        if coordinate is not None:
            positions[index] = coordinate
            mask[index] = True
    if not np.any(mask):
        raise ValueError(
            "None of the requested motif atoms exist in AF3's dense atom layout. "
            "Check the target chain and residue mapping."
        )
    positions[mask] -= positions[mask].mean(axis=0)
    return positions, mask


def build_protenix_projection(spec: MotifSpec, atom_array) -> tuple[np.ndarray, np.ndarray]:
    """Map motif atoms to the flattened atom order used by Protenix."""
    coordinates = spec.source_coordinates()
    positions = np.zeros((len(atom_array), 3), dtype=np.float32)
    mask = np.zeros(len(atom_array), dtype=bool)
    for index, (chain_id, residue_id, atom_name) in enumerate(
        zip(atom_array.chain_id, atom_array.res_id, atom_array.atom_name)
    ):
        coordinate = coordinates.get((str(chain_id), int(residue_id), str(atom_name).strip()))
        if coordinate is not None:
            positions[index] = coordinate
            mask[index] = True
    if not np.any(mask):
        raise ValueError(
            "None of the requested motif atoms exist in Protenix's atom layout. "
            "Check the target chain and residue mapping."
        )
    positions[mask] -= positions[mask].mean(axis=0)
    return positions, mask


def inject_protenix_motif_sequence(input_json: list[dict[str, Any]], spec: MotifSpec) -> None:
    """Preserve motif residues in a Protenix input using its entity-order chains."""
    if not input_json or not isinstance(input_json[0], Mapping):
        raise ValueError("Protenix input must be a non-empty JSON list.")
    target_chain = spec.target_chain
    source_sequence = spec.source_residue_sequence()
    protein_index = 0
    for entry in input_json[0].get("sequences", []):
        protein = entry.get("proteinChain")
        if protein is None:
            continue
        count = int(protein.get("count", 1))
        chain_ids = string.ascii_uppercase[protein_index:protein_index + count]
        protein_index += count
        if target_chain not in chain_ids:
            continue
        sequence = list(protein.get("sequence", ""))
        for mapping in spec.mappings:
            mapped_chain, target_res_id, _ = mapping.target
            if mapped_chain == target_chain:
                if target_res_id < 1 or target_res_id > len(sequence):
                    raise ValueError(
                        f"Motif target residue {target_chain}:{target_res_id} is outside "
                        "the Protenix query sequence."
                    )
                sequence[target_res_id - 1] = source_sequence[(mapping.source[0], mapping.source[1])]
        protein["sequence"] = "".join(sequence)
        return
    raise ValueError(f"Protenix JSON has no protein chain {target_chain!r} for motif.")


def inject_af3_motif_template(
    input_json: dict[str, Any], spec: MotifSpec, include_template: bool = True
) -> None:
    """Preserve motif sequence and optionally prepend an AF3 template."""
    target_chain = spec.target_chain
    protein_entry = None
    for entry in input_json.get("sequences", []):
        protein = entry.get("protein")
        if not protein:
            continue
        chain_ids = protein.get("id", [])
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]
        if target_chain in chain_ids:
            if protein_entry is not None:
                raise ValueError(f"AF3 JSON contains duplicate chain {target_chain!r}.")
            protein_entry = protein
    if protein_entry is None:
        raise ValueError(f"AF3 JSON has no protein chain {target_chain!r} for motif.")

    sequence = list(protein_entry.get("sequence", ""))
    source_sequence = spec.source_residue_sequence()
    for mapping in spec.mappings:
        target_chain_id, target_res_id, _ = mapping.target
        if target_chain_id != target_chain:
            continue
        if not 1 <= target_res_id <= len(sequence):
            raise ValueError(
                f"Motif target residue {target_chain}:{target_res_id} is outside "
                "the AF3 query sequence."
            )
        source_key = (mapping.source[0], mapping.source[1])
        sequence[target_res_id - 1] = source_sequence[source_key]
    protein_entry["sequence"] = "".join(sequence)

    for msa_key in ("unpairedMsa", "pairedMsa"):
        msa = protein_entry.get(msa_key)
        if not msa:
            continue
        lines = str(msa).strip().splitlines()
        if lines and lines[0].startswith(">"):
            first_record_end = next(
                (index for index, line in enumerate(lines[1:], start=1)
                 if line.startswith(">")),
                len(lines),
            )
            lines[1:first_record_end] = [protein_entry["sequence"]]
            protein_entry[msa_key] = "\n".join(lines) + "\n"

    if not include_template:
        return

    template = spec.build_af3_template(len(sequence))

    templates = protein_entry.get("templates")
    if templates is None:
        templates = []
        protein_entry["templates"] = templates
    if not any(
        existing.get("queryIndices") == template["queryIndices"]
        and existing.get("templateIndices") == template["templateIndices"]
        and existing.get("mmcif") == template["mmcif"]
        for existing in templates
    ):
        # AF3 keeps only the first max_templates entries during featurisation.
        # Put the user-specified motif first so an existing template list cannot
        # silently hide the motif constraint.
        templates[:] = [template] + list(templates)
