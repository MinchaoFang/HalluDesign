

def find_target_heavyatoms(pose, target_resno):
    tgt_res = pose.residue(target_resno)
    print(f"Finding target heavyatoms for residue {target_resno}")
    target_heavyatoms = []
    for n in range(1, tgt_res.natoms()+1):
        if tgt_res.atom_type(n).is_heavyatom():
            target_heavyatoms.append(n)
    return target_heavyatoms

def get_packer_layers(pose, ref_resno, cuts, target_atoms, do_not_design=None, allow_design=None, design_GP=False):
    """
    Finds residues that are within certain distances from target atoms, defined though <cuts>.
    Returns a list of lists where each embedded list contains residue numbers belongin to that layer.
    Last list contains all other residue numbers that were left over.
    get_packer_layers(pose, target_atoms, do_not_design, cuts) -> list
    Arguments:
        pose (object, pyrosetta.rosetta.core.pose.Pose)
        target_atoms (list) :: list of integers.
                               Atom numbers of a residue or a ligand that are used to calculate residue distances.
        do_not_design (list) :: list of integers. Residue numbers that are not allowed to be designed.
                                Used to override layering to not design matched residues.
        cuts (list) :: list of floats.
        design_GP (bool) :: if True, allows redesign of GLY and PRO
    """
    assert len(cuts) > 3, f"Not enough layer cut distances defined {cuts}"

    if do_not_design is None:
        do_not_design = []
    if allow_design is None:
        allow_design = []

    KEEP_RES = ["GLY", "PRO"]
    if design_GP is True:
        KEEP_RES = []

    residues = []
    ligand_atoms = []
    if isinstance(target_atoms, list):
        for a in target_atoms:
            ligand_atoms.append(pose.residue(ref_resno).xyz(a))
    if isinstance(target_atoms, dict):
        for k in target_atoms:
            for a in target_atoms[k]:
                ligand_atoms.append(pose.residue(k).xyz(a))

    residues = [[] for i in cuts] + [[]]

    for resno in range(1, pose.size()):
        if pose.residue(resno).is_ligand():
            continue
        if pose.residue(resno).is_virtual_residue():
            continue
        if resno in do_not_design:
            # do_not_design residues are added to repack layer
            residues[2].append(resno)
            continue
        if resno in allow_design:
            # allow_design residues are added to design layer
            residues[0].append(resno)
            continue
        resname = pose.residue(resno).name3()
        CA = pose.residue(resno).xyz('CA')
        CA_distances = []

        if 'GLY' not in resname:
            CB = pose.residue(resno).xyz('CB')
            CB_distances = []

        for a in ligand_atoms:
            CA_distances.append((a - CA).norm())
            if 'GLY' not in resname:
                CB_distances.append((a - CB).norm())
        CA_mindist = min(CA_distances)

        if 'GLY' not in resname:
            CB_mindist = min(CB_distances)

        # Figuring out into which cut that residue belongs to,
        # based on the smallest CA distance and whether the CA is further away from the ligand than CB or not.
        # PRO and GLY are disallowed form the first two cuts that would allow design,
        # they can only be repacked.

        if CA_mindist <= cuts[0] and resname not in KEEP_RES:
            residues[0].append(resno)
        elif CA_mindist <= cuts[1] and resname not in KEEP_RES:
            if resname == "GLY":
                residues[1].append(resno)
            elif CB_mindist < CA_mindist:
                residues[1].append(resno)
            elif CA_mindist < cuts[1]-1.0 and CB_mindist < cuts[1]-1.0:
                # Special case when the residue is close, but CB is pointing slightly away.
                residues[1].append(resno)
            else:
                residues[2].append(resno)
        elif CA_mindist <= cuts[2]:
            residues[2].append(resno)
        elif CA_mindist <= cuts[3] and resname not in KEEP_RES:
            if resname == "GLY":
                residues[3].append(resno)
            elif CB_mindist < CA_mindist:
                residues[3].append(resno)
            else:
                residues[-1].append(resno)
        else:
            residues[-1].append(resno)

    return residues

def get_residues_with_close_sc(pose, target_resno, target_atoms, residues=None, cutoff_sc=None, exclude_residues=None):
    """
    """
    if residues is None:
        residues = [x for x in range(1, pose.size()+1)]
    if exclude_residues is None:
        exclude_residues = []

    if cutoff_sc is None:
        cutoff_sc = 4.5

    target_res = pose.residue(target_resno)
    close_ones = []
    for resno in residues:
        if resno in exclude_residues:
            continue
        if pose.residue(resno).is_ligand():
            continue
        if (pose.residue(resno).nbr_atom_xyz() - target_res.nbr_atom_xyz()).norm() > 20.0:
            continue
        res = pose.residue(resno)
        close_enough = False
        for atomno in range(1, res.natoms()):
            if not res.atom_type(atomno).is_heavyatom():
                continue
            for ha in target_atoms:
                if (res.xyz(atomno) - target_res.xyz(ha)).norm() < cutoff_sc:
                    close_enough = True
                    close_ones.append(resno)
                    break
                if close_enough is True:
                    break
            if close_enough is True:
                break
    return close_ones

def find_pocket_residues_based_on_rosetta(pdbfile=None, pose=None, target_resno=None, cutoff_CA=5.5, cutoff_sc=4.5):
    """
    Finds residues that should be fixed during MPNN design based on proximity to target residues or ligands.

    Parameters

    ----------
    pdbfile : str, optional

        Path to the PDB file.
    pose : pyrosetta.rosetta.core.pose.Pose, optional

        Rosetta pose object.
    target_resno : int or list, optional

        Target residue number(s). If None, all ligands will be considered.
    cutoff_CA : float, optional

        Cutoff distance for CA atoms to target heavy atoms. Default is 5.5.
    cutoff_sc : float, optional

        Cutoff distance for side chain atoms to target heavy atoms. Default is 4.5.

    Returns

    -------
    fixed_residues : list

        List of residue numbers that should be fixed.
    """
    assert isinstance(pdbfile, (type(None), str)), "pdbfile must be a string or None"
    assert isinstance(target_resno, (type(None), int, list)), "target_resno must be an int, list, or None"
    assert not all([x is None for x in [pdbfile, pose]]), "Must provide either a path to a pdbfile or a pose object as input"
    try:
        if isinstance(target_resno, int):
            target_resno = [target_resno]

        if pose is None:
            pose = pyr.pose_from_file(pdbfile)

        cuts = [cutoff_CA, cutoff_CA + 2.5, cutoff_CA + 4.5, cutoff_CA + 6.5]

        tgt_residues = []
        if target_resno is None:
            for r in pose.residues:
                if r.is_ligand() and not r.is_virtual_residue():
                    tgt_residues.append(r.seqpos())
        else:
            for r in pose.residues:
                if r.seqpos() in target_resno:
                    if not r.is_ligand():
                        print(f"Warning! Residue {r.name3()}-{r.seqpos()} is not a ligand! I hope this is intentional.")
                    tgt_residues.append(r.seqpos())

        pocket_residues = []
        for tgt_resno in tgt_residues:
            heavyatoms = find_target_heavyatoms(pose, tgt_resno)
            layers = get_packer_layers(pose, tgt_resno, cuts=cuts, target_atoms=heavyatoms, design_GP=True)
            close_ones = get_residues_with_close_sc(pose, tgt_resno, heavyatoms, cutoff_sc=cutoff_sc, exclude_residues=[])
            pocket_residues += layers[0] + close_ones

        pocket_residues = list(set(pocket_residues))
        pdb_info = pose.pdb_info()
        fixed_residues =[]
        for resno in pocket_residues:
            chain = pdb_info.chain(resno)
            pdb_num = pdb_info.number(resno)
            res_id = f"{chain}{pdb_num}"
            fixed_residues.append(res_id)
        protein_length  = sum(1 for res in pose.residues if not res.is_ligand() and not res.is_virtual_residue())
        if len(pocket_residues) == protein_length:
            print("The number of residues equals the protein's length. Using alternative calculation method.")
            fixed_residues = find_pocket_residues_based_on_distance(pdbfile=pdbfile ,cutoff=8.0)
    except Exception as e:
        print(f"Error in pocket res calculation method: {e}")
        fixed_residues = find_pocket_residues_based_on_distance(pdbfile=pdbfile ,cutoff=8.0)
        
    return fixed_residues