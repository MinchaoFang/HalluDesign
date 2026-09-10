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

_CONSTRAINT_MODES = frozenset({
    "template",
    "af3_projection",
    "af3_soft_projection",
    "protenix_projection",
    "protenix_soft_projection",
})
_MOTIF_MODES = frozenset({"off", "refine", "scaffold_then_refine"})
_BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


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
    atom = parts[2].strip().upper() if len(parts) == 3 else ""
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
                    key = (
                        str(chain.id).strip(),
                        res_id,
                        str(atom.name).strip().upper(),
                    )
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
    soft_projection_weight: float = 0.15
    noisy_projection_weight: float = 0.15
    denoising_projection_weight: float = 0.15
    x0_projection_weight: float = 0.0
    smooth_neighbor_residues: int = 0
    sequence_fixed_targets: tuple[tuple[str, int], ...] = ()

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
        # Keep the old single-weight field as a compatibility fallback. New
        # specs can control the two projection sites independently.
        soft_projection_weight = float(raw.get("soft_projection_weight", 0.15))
        noisy_projection_weight = float(
            raw.get("noisy_projection_weight", soft_projection_weight)
        )
        denoising_projection_weight = float(
            raw.get("denoising_projection_weight", soft_projection_weight)
        )
        # x0 guidance is a separate sampler feature.  It defaults to zero so
        # older motif specs retain exactly their previous behavior.
        x0_projection_weight = float(raw.get("x0_projection_weight", 0.0))
        smooth_neighbor_residues = int(raw.get("smooth_neighbor_residues", 0))
        if scaffold_cycles < 0:
            raise ValueError("scaffold_cycles must be non-negative.")
        if scaffold_steps <= 0 or refine_steps <= 0:
            raise ValueError("scaffold_steps and refine_steps must be positive.")
        if not 0.0 <= soft_projection_weight <= 1.0:
            raise ValueError("soft_projection_weight must be between 0 and 1.")
        if not 0.0 <= noisy_projection_weight <= 1.0:
            raise ValueError("noisy_projection_weight must be between 0 and 1.")
        if not 0.0 <= denoising_projection_weight <= 1.0:
            raise ValueError(
                "denoising_projection_weight must be between 0 and 1."
            )
        if not 0.0 <= x0_projection_weight <= 1.0:
            raise ValueError("x0_projection_weight must be between 0 and 1.")
        if smooth_neighbor_residues < 0:
            raise ValueError("smooth_neighbor_residues must be non-negative.")
        if {"af3_projection", "af3_soft_projection"}.issubset(constraint_modes):
            raise ValueError(
                "Use either af3_projection or af3_soft_projection, not both."
            )
        if {"protenix_projection", "protenix_soft_projection"}.issubset(constraint_modes):
            raise ValueError(
                "Use either protenix_projection or protenix_soft_projection, not both."
            )
        target_atoms = [mapping.target for mapping in mappings]
        if len(target_atoms) != len(set(target_atoms)):
            raise ValueError("residue_mapping contains duplicate target atoms.")
        raw_sequence_fixed = raw.get("sequence_fixed_targets")
        if raw_sequence_fixed is None:
            sequence_fixed_targets = tuple(sorted({mapping.target[:2] for mapping in mappings}))
        else:
            if isinstance(raw_sequence_fixed, str):
                raw_sequence_fixed = [raw_sequence_fixed]
            if not isinstance(raw_sequence_fixed, (list, tuple, set)):
                raise ValueError("sequence_fixed_targets must be a list of chain:residue values.")
            sequence_fixed_targets = tuple(
                sorted({_parse_atom_key(value)[:2] for value in raw_sequence_fixed})
            )
            mapped_targets = {mapping.target[:2] for mapping in mappings}
            unknown_targets = set(sequence_fixed_targets) - mapped_targets
            if unknown_targets:
                raise ValueError(
                    "sequence_fixed_targets contains residues without motif mappings: "
                    f"{sorted(unknown_targets)}"
                )
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
            soft_projection_weight=soft_projection_weight,
            noisy_projection_weight=noisy_projection_weight,
            denoising_projection_weight=denoising_projection_weight,
            x0_projection_weight=x0_projection_weight,
            smooth_neighbor_residues=smooth_neighbor_residues,
            sequence_fixed_targets=sequence_fixed_targets,
        )

    def with_runtime_options(
        self,
        motif_mode: str | None = None,
        scaffold_steps: int | None = None,
        refine_steps: int | None = None,
        soft_projection_weight: float | None = None,
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
        if soft_projection_weight is not None:
            if not 0.0 <= soft_projection_weight <= 1.0:
                raise ValueError(
                    "soft_projection_weight must be between 0 and 1."
                )
            # The legacy CLI option remains a convenient way to set both
            # sites. Independent weights are intentionally JSON-only.
            updates["soft_projection_weight"] = soft_projection_weight
            updates["noisy_projection_weight"] = soft_projection_weight
            updates["denoising_projection_weight"] = soft_projection_weight
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

    def _source_coordinates(self) -> dict[tuple[str, int, str], np.ndarray]:
        """Return selected native motif atoms keyed by source identifiers."""
        source_atoms = _structure_atom_index(self.motif_pdb)
        result: dict[tuple[str, int, str], np.ndarray] = {}
        missing = []
        for mapping in self.mappings:
            allowed_atoms = self._allowed_atoms_for_mapping(mapping)
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
                elif source_key in result and not np.allclose(result[source_key], coord):
                    raise ValueError(f"Conflicting source atom coordinates: {source_key}")
                else:
                    result[source_key] = coord
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
                    source_atom_key = (source_key[0], source_key[1], atom_name)
                    result[source_atom_key] = coord
        if missing:
            raise ValueError(f"Motif atoms/residues were not found: {missing[:8]}")
        return result

    def source_coordinates(self) -> dict[tuple[str, int, str], np.ndarray]:
        """Return selected native motif atoms keyed by target identifiers."""
        source_atoms = self._source_coordinates()
        result: dict[tuple[str, int, str], np.ndarray] = {}
        for mapping in self.mappings:
            source_residue = mapping.source[:2]
            target_residue = mapping.target[:2]
            allowed_atoms = self._allowed_atoms_for_mapping(mapping)
            if mapping.source[2]:
                atom_names = (mapping.source[2],)
            else:
                atom_names = tuple(
                    atom_name
                    for chain, residue_id, atom_name in source_atoms
                    if (chain, residue_id) == source_residue
                    and (allowed_atoms is None or atom_name in allowed_atoms)
                )
            for atom_name in atom_names:
                source_key = (*source_residue, atom_name)
                coordinate = source_atoms.get(source_key)
                if coordinate is None:
                    continue
                target_key = (*target_residue, atom_name)
                if target_key in result and not np.allclose(result[target_key], coordinate):
                    raise ValueError(f"Duplicate target atom mapping: {target_key}")
                result[target_key] = coordinate
        if not result:
            raise ValueError("The motif mapping did not select any atoms.")
        return result

    def _allowed_atoms_for_mapping(self, mapping: MotifMapping):
        """Return the atom policy for one mapped residue.

        A residue whose identity is masked cannot safely keep reference
        side-chain coordinates: its random amino-acid type may have a
        different atom topology.  Backbone coordinates remain valid.  Specs
        without masked targets retain the original global policy.
        """
        if self.fixed_atom_policy == "backbone":
            return _BACKBONE_ATOMS
        if self.fixed_atom_policy == "custom":
            return set(self.fixed_atom_names)
        if mapping.target[:2] not in self.sequence_fixed_targets:
            return _BACKBONE_ATOMS
        return None

    def target_residue_strings(self) -> list[str]:
        return [f"{chain}{res_id}" for chain, res_id in self.sequence_fixed_targets]

    @property
    def target_chains(self) -> set[str]:
        return {chain for chain, _ in self.target_residues}

    @property
    def target_chain(self) -> str:
        chains = {chain for chain, _ in self.target_residues}
        if len(chains) != 1:
            raise ValueError(
                "An AF3 motif template must target exactly one protein chain; "
                f"got {sorted(chains)}."
            )
        return next(iter(chains))

    def for_target_chain(self, target_chain: str) -> "MotifSpec":
        """Return the subset of this motif that maps to one target chain."""
        mappings = tuple(
            mapping for mapping in self.mappings
            if mapping.target[0] == target_chain
        )
        if not mappings:
            raise ValueError(f"Motif has no mappings for target chain {target_chain!r}.")
        mapped_residues = {mapping.target[:2] for mapping in mappings}
        sequence_fixed_targets = tuple(
            target for target in self.sequence_fixed_targets
            if target in mapped_residues
        )
        return replace(
            self,
            mappings=mappings,
            sequence_fixed_targets=sequence_fixed_targets,
        )

    def _template_structure(self):
        """Return the mapped motif as one chain with contiguous residue IDs."""
        source_chains = {chain for chain, _ in self.source_residues}
        if len(source_chains) != 1:
            raise ValueError(
                "An AF3 motif template must contain exactly one source chain; "
                f"got {sorted(source_chains)}."
            )
        source_chain_id = next(iter(source_chains))
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
            residue.detach_parent()
            residue.id = (" ", template_index, " ")
            target_residues = {
                mapping.target[:2]
                for mapping in self.mappings
                if mapping.source[:2] == source_key
            }
            if not target_residues:
                raise ValueError(f"Motif residue has no target mapping: {source_key}")
            if self.fixed_atom_policy == "backbone":
                allowed_atoms = _BACKBONE_ATOMS
            elif self.fixed_atom_policy == "custom":
                allowed_atoms = set(self.fixed_atom_names)
            elif not any(
                target in self.sequence_fixed_targets for target in target_residues
            ):
                allowed_atoms = _BACKBONE_ATOMS
            else:
                allowed_atoms = None
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


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return row-vector rotation and translation mapping source onto target."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = source_centered.T @ target_centered
    left, _, right_transposed = np.linalg.svd(covariance)
    rotation = left @ right_transposed
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transposed
    translation = target_center - source_center @ rotation
    return rotation, translation


def calculate_motif_backbone_rmsd(
    motif_spec: MotifSpec | str,
    predicted_path: str,
) -> float:
    """Return Kabsch-aligned backbone RMSD for one mapped motif structure.

    This helper is intentionally independent of the diffusion sampler.  It is
    used by the optional multi-start orchestration to rank random-init
    structures before formal scaffolding.  A motif spec with a ``backbone``
    atom policy still produces the same backbone score; side-chain policy does
    not affect which backbone atoms are selected.
    """
    if isinstance(motif_spec, str):
        motif_spec = MotifSpec.from_file(motif_spec)

    source_atoms = _structure_atom_index(motif_spec.motif_pdb)
    predicted = _structure_atom_index(predicted_path)
    reference_coords = []
    predicted_coords = []
    for mapping in motif_spec.mappings:
        source_residue = mapping.source[:2]
        target_residue = mapping.target[:2]
        atom_names = (
            (mapping.source[2],)
            if mapping.source[2]
            else tuple(_BACKBONE_ATOMS)
        )
        for atom_name in atom_names:
            if atom_name not in _BACKBONE_ATOMS:
                continue
            source_coord = source_atoms.get((*source_residue, atom_name))
            target_coord = predicted.get((*target_residue, atom_name))
            if source_coord is None or target_coord is None:
                continue
            reference_coords.append(np.asarray(source_coord, dtype=np.float64))
            predicted_coords.append(np.asarray(target_coord, dtype=np.float64))

    if len(reference_coords) < 3:
        raise ValueError(
            "Fewer than three shared motif backbone atoms are available for RMSD."
        )
    rotation, translation = _rigid_transform(
        np.asarray(predicted_coords, dtype=np.float64),
        np.asarray(reference_coords, dtype=np.float64),
    )
    aligned = np.asarray(predicted_coords, dtype=np.float64) @ rotation + translation
    reference = np.asarray(reference_coords, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1))))


def _mapped_motif_coordinates(
    spec: MotifSpec,
    current_structure_path: str | None = None,
) -> dict[tuple[str, int, str], np.ndarray]:
    """Map native motif atoms into the current target structure frame.

    When a current structure is supplied, the native motif is fitted to the
    mapped target residues.  This is deliberately based on the current target
    coordinates rather than an independently centered motif, so chain breaks
    in the current scaffold are not replaced by absolute native coordinates.
    """
    source_coordinates = spec._source_coordinates()
    native_coordinates = _structure_atom_index(spec.motif_pdb)

    def remap_to_targets(coordinates):
        target_coordinates = {}
        seen_target_atoms = set()
        for mapping in spec.mappings:
            source_residue = mapping.source[:2]
            target_residue = mapping.target[:2]
            allowed_atoms = spec._allowed_atoms_for_mapping(mapping)
            if mapping.source[2]:
                atom_names = (mapping.source[2],)
            else:
                atom_names = tuple(
                    atom_name
                    for chain, residue_id, atom_name in coordinates
                    if (chain, residue_id) == source_residue
                    and (allowed_atoms is None or atom_name in allowed_atoms)
                )
            for atom_name in atom_names:
                source_key = (*source_residue, atom_name)
                coord = coordinates.get(source_key)
                if coord is None:
                    continue
                target_key = (*target_residue, atom_name)
                if target_key in seen_target_atoms and not np.allclose(
                    target_coordinates[target_key], coord
                ):
                    raise ValueError(f"Duplicate target atom mapping: {target_key}")
                target_coordinates[target_key] = coord
                seen_target_atoms.add(target_key)
        return target_coordinates

    if not current_structure_path:
        # The sampler will rigidly fit these native coordinates to its current
        # motif atoms before every projection step. No absolute frame is needed.
        return remap_to_targets(source_coordinates)

    current_coordinates = _structure_atom_index(current_structure_path)
    fit_source = []
    fit_target = []
    seen_residues = set()
    # CA anchors are residue-level and avoid overweighting long side chains.
    for mapping in spec.mappings:
        source_residue = mapping.source[:2]
        target_residue = mapping.target[:2]
        if source_residue in seen_residues:
            continue
        source_ca = native_coordinates.get((*source_residue, "CA"))
        target_ca = current_coordinates.get((*target_residue, "CA"))
        if source_ca is not None and target_ca is not None:
            fit_source.append(source_ca)
            fit_target.append(target_ca)
            seen_residues.add(source_residue)

    # For very short motifs, use all shared mapped backbone atoms if possible.
    if len(fit_source) < 3:
        fit_source = []
        fit_target = []
        for mapping in spec.mappings:
            source_residue = mapping.source[:2]
            target_residue = mapping.target[:2]
            for atom_name in _BACKBONE_ATOMS:
                source_coord = native_coordinates.get((*source_residue, atom_name))
                target_coord = current_coordinates.get((*target_residue, atom_name))
                if source_coord is not None and target_coord is not None:
                    fit_source.append(source_coord)
                    fit_target.append(target_coord)

    if fit_source:
        source_fit = np.asarray(fit_source, dtype=np.float64)
        target_fit = np.asarray(fit_target, dtype=np.float64)
        if len(source_fit) >= 3:
            rotation, translation = _rigid_transform(source_fit, target_fit)
        else:
            rotation = np.eye(3, dtype=np.float64)
            translation = target_fit.mean(axis=0) - source_fit.mean(axis=0)
    else:
        raise ValueError(
            "No shared motif anchor atoms were found between the native motif "
            f"and current structure {current_structure_path}."
        )

    aligned = {
        key: np.asarray(coord, dtype=np.float32) @ rotation.astype(np.float32)
        + translation.astype(np.float32)
        for key, coord in source_coordinates.items()
    }
    # Preserve the current scaffold frame while removing arbitrary global
    # translation.  This is not motif-centering: the center comes from the
    # complete current structure.
    current_center = np.mean(np.asarray(list(current_coordinates.values())), axis=0)
    aligned = {key: coord - current_center for key, coord in aligned.items()}
    return remap_to_targets(aligned)


def build_af3_dense_projection(
    spec: MotifSpec,
    token_atoms_layout,
    current_structure_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map aligned motif atoms to AF3's actual dense atom layout."""
    coordinates = _mapped_motif_coordinates(spec, current_structure_path)
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
    return positions, mask


def build_protenix_projection(
    spec: MotifSpec,
    atom_array,
    current_structure_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map aligned motif atoms to the flattened atom order used by Protenix."""
    coordinates = _mapped_motif_coordinates(spec, current_structure_path)
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
    return positions, mask


def _build_smoothing_metadata(
    spec: MotifSpec,
    chain_ids,
    residue_ids,
    atom_names,
    motif_mask,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a tapered displacement field around the mapped motif.

    Neighbor residues never receive fabricated reference coordinates.  They
    inherit the displacement of the closest motif boundary atom, which keeps
    their internal geometry intact while allowing the motif/scaffold boundary
    to move smoothly.  Sequence adjacency is inferred only from residues that
    are present in the model layout and have consecutive residue IDs.
    """
    shape = np.asarray(motif_mask).shape
    flat_chains = np.asarray(chain_ids).reshape(-1)
    flat_residues = np.asarray(residue_ids).reshape(-1)
    flat_names = np.asarray(atom_names).reshape(-1)
    flat_motif_mask = np.asarray(motif_mask, dtype=bool).reshape(-1)
    if not (
        flat_chains.shape == flat_residues.shape
        == flat_names.shape
        == flat_motif_mask.shape
    ):
        raise ValueError("Motif smoothing metadata inputs must have the same shape.")

    weights = np.zeros(flat_motif_mask.shape, dtype=np.float32)
    source_indices = np.zeros(flat_motif_mask.shape, dtype=np.int32)
    radius = spec.smooth_neighbor_residues
    if radius == 0:
        return weights.reshape(shape), source_indices.reshape(shape)

    residue_atoms: dict[tuple[str, int], list[int]] = {}
    for index, (chain, residue, atom_name) in enumerate(
        zip(flat_chains, flat_residues, flat_names)
    ):
        if not str(atom_name).strip():
            continue
        residue_atoms.setdefault((str(chain), int(residue)), []).append(index)

    motif_residues = {
        residue for residue in spec.target_residues if residue in residue_atoms
    }
    if not motif_residues:
        return weights.reshape(shape), source_indices.reshape(shape)

    motif_atom_indices: dict[tuple[str, int], list[int]] = {}
    for residue, indices in residue_atoms.items():
        selected = [index for index in indices if flat_motif_mask[index]]
        if selected:
            motif_atom_indices[residue] = selected

    def boundary_atom(residue: tuple[str, int]) -> int | None:
        candidates = motif_atom_indices.get(residue, [])
        if not candidates:
            return None
        preferred = {"CA": 0, "N": 1, "C": 2, "O": 3}
        return min(
            candidates,
            key=lambda index: (preferred.get(str(flat_names[index]).strip(), 4), index),
        )

    # Multi-source BFS on each chain.  A residue-index gap is a hard barrier.
    distances: dict[tuple[str, int], tuple[int, tuple[str, int]]] = {}
    frontier = []
    for residue in sorted(motif_residues):
        distances[residue] = (0, residue)
        frontier.append(residue)
    while frontier:
        current = frontier.pop(0)
        distance, source_residue = distances[current]
        if distance >= radius:
            continue
        chain, residue_id = current
        for neighbor_id in (residue_id - 1, residue_id + 1):
            neighbor = (chain, neighbor_id)
            if neighbor not in residue_atoms:
                continue
            candidate = (distance + 1, source_residue)
            previous = distances.get(neighbor)
            if previous is None or candidate < previous:
                distances[neighbor] = candidate
                frontier.append(neighbor)

    source_atom_by_residue = {
        residue: boundary_atom(residue) for residue in motif_residues
    }
    for residue, (distance, source_residue) in distances.items():
        source_index = source_atom_by_residue.get(source_residue)
        if source_index is None:
            continue
        alpha = 1.0 if distance == 0 else (radius + 1 - distance) / (radius + 1)
        for index in residue_atoms[residue]:
            if distance == 0:
                if flat_motif_mask[index]:
                    weights[index] = 1.0
                    source_indices[index] = index
            else:
                weights[index] = np.float32(alpha)
                source_indices[index] = source_index

    return weights.reshape(shape), source_indices.reshape(shape)


def build_af3_smoothing_metadata(
    spec: MotifSpec,
    token_atoms_layout,
    motif_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build neighbor smoothing metadata in AF3's dense atom layout."""
    return _build_smoothing_metadata(
        spec,
        token_atoms_layout.chain_id,
        token_atoms_layout.res_id,
        token_atoms_layout.atom_name,
        motif_mask,
    )


def build_protenix_smoothing_metadata(
    spec: MotifSpec,
    atom_array,
    motif_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build neighbor smoothing metadata in Protenix's flattened atom order."""
    return _build_smoothing_metadata(
        spec,
        atom_array.chain_id,
        atom_array.res_id,
        atom_array.atom_name,
        motif_mask,
    )


def inject_protenix_motif_sequence(input_json: list[dict[str, Any]], spec: MotifSpec) -> None:
    """Preserve motif residues in a Protenix input using its entity-order chains."""
    if not input_json or not isinstance(input_json[0], Mapping):
        raise ValueError("Protenix input must be a non-empty JSON list.")
    source_sequence = spec.source_residue_sequence()
    chain_index = 0
    matched_chains = set()
    for entry in input_json[0].get("sequences", []):
        entity = next(iter(entry.values()), None)
        if not isinstance(entity, Mapping):
            continue
        count = int(entity.get("count", 1))
        if count < 1:
            raise ValueError("Protenix entity count must be positive.")
        chain_ids = string.ascii_uppercase[chain_index:chain_index + count]
        if len(chain_ids) != count:
            raise ValueError(
                "Protenix input contains more chains than the supported A-Z "
                "motif mapping convention."
            )
        chain_index += count
        protein = entry.get("proteinChain")
        if protein is None:
            continue
        if any(chain_id in spec.target_chains for chain_id in chain_ids) and count != 1:
            raise ValueError(
                "Motif targets a protein entity with count > 1. Split repeated "
                "copies into separate proteinChain entries before applying a "
                "chain-specific motif."
            )
        for chain_id in chain_ids:
            if chain_id not in spec.target_chains:
                continue
            matched_chains.add(chain_id)
            sequence = list(protein.get("sequence", ""))
            for mapping in spec.mappings:
                mapped_chain, target_res_id, _ = mapping.target
                if (
                    mapped_chain == chain_id
                    and (mapped_chain, target_res_id) in spec.sequence_fixed_targets
                ):
                    if target_res_id < 1 or target_res_id > len(sequence):
                        raise ValueError(
                            f"Motif target residue {mapped_chain}:{target_res_id} is outside "
                            "the Protenix query sequence."
                        )
                    sequence[target_res_id - 1] = source_sequence[(mapping.source[0], mapping.source[1])]
            protein["sequence"] = "".join(sequence)
    missing_chains = spec.target_chains - matched_chains
    if missing_chains:
        raise ValueError(
            f"Protenix JSON has no protein chain(s) {sorted(missing_chains)} for motif."
        )


def inject_af3_motif_template(
    input_json: dict[str, Any], spec: MotifSpec, include_template: bool = True
) -> None:
    """Preserve motif sequences and optionally prepend per-chain AF3 templates."""
    protein_entries = {}
    for entry in input_json.get("sequences", []):
        protein = entry.get("protein")
        if not protein:
            continue
        chain_ids = protein.get("id", [])
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]
        for chain_id in chain_ids:
            if chain_id in protein_entries:
                raise ValueError(f"AF3 JSON contains duplicate chain {chain_id!r}.")
            protein_entries[chain_id] = protein

    missing_chains = spec.target_chains - set(protein_entries)
    if missing_chains:
        raise ValueError(
            f"AF3 JSON has no protein chain(s) {sorted(missing_chains)} for motif."
        )

    source_sequence = spec.source_residue_sequence()
    for target_chain in sorted(spec.target_chains):
        chain_spec = spec.for_target_chain(target_chain)
        protein_entry = protein_entries[target_chain]
        sequence = list(protein_entry.get("sequence", ""))
        for mapping in chain_spec.mappings:
            _, target_res_id, _ = mapping.target
            if (target_chain, target_res_id) not in chain_spec.sequence_fixed_targets:
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
            continue

        template = chain_spec.build_af3_template(len(sequence))
        templates = protein_entry.setdefault("templates", [])
        if not any(
            existing.get("queryIndices") == template["queryIndices"]
            and existing.get("templateIndices") == template["templateIndices"]
            and existing.get("mmcif") == template["mmcif"]
            for existing in templates
        ):
            # AF3 keeps only the first max_templates entries during featurisation.
            templates[:] = [template] + list(templates)
