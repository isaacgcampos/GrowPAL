import time
import numpy as np
from ase import Atom
from multiprocessing import Pool, cpu_count
from aegon.libutils import rename, adjacency_matrix, sort_by_energy
from molsympy.api import get_inequivalent
#------------------------------------------------------------------------------------------
def neighbor_finder(atoms, dtol=1.2):
    """Build an adjacency dictionary mapping each atom index to its bonded neighbor indices.

    in:
        atoms (ase.Atoms): cluster structure to analyze.
        dtol (float): distance scale factor relative to sum of covalent radii for bond
            detection (default 1.2).
    out:
        dict: mapping of atom index (int) to list of neighbor indices (list of int).
    """
    adj_mtx = adjacency_matrix(atoms, dtol)
    return {i: list(np.nonzero(row)[0]) for i, row in enumerate(adj_mtx)}

#------------------------------------------------------------------------------------------
def focused_expansion(molin, vector):
    """Expand all atoms radially away from a given reference position using an exponential displacement.

    in:
        molin (ase.Atoms): structure to expand.
        vector (numpy.ndarray): reference position from which atoms are pushed outward.
    out:
        ase.Atoms: copy of the structure with all atoms displaced radially outward from vector.
    """
    molout = molin.copy()
    positions = molout.positions
    vectors = positions - vector
    distances = np.linalg.norm(vectors, axis=1, keepdims=True)

    distances = np.where(distances == 0, 1e-10, distances)

    rnorm = vectors / distances
    edel = np.exp(-distances) * 1.2
    new_positions = positions + rnorm * edel

    molout.positions = new_positions
    return molout

#------------------------------------------------------------------------------------------
def add_all_interstitial_atoms(original_molecule, neighbors_dict, specie):
    """Add candidate atoms at the centroid of every triangle in the adjacency graph.

    Traverses each edge (a, n) with a < n exactly once and looks for common
    neighbours n2 > n, guaranteeing that every triangle is visited exactly once
    without any bookkeeping set.

    in:
        original_molecule (ase.Atoms): base cluster structure.
        neighbors_dict (dict): adjacency dictionary from neighbor_finder.
        specie (str): chemical symbol of the new atom to insert at each triangle centroid.
    out:
        ase.Atoms: extended structure with one new atom added per unique triangle.
    """
    xxx_mol       = original_molecule.copy()
    all_positions = original_molecule.positions
    # precompute neighbour sets once to enable O(1) membership tests
    neighbor_sets = {i: set(v) for i, v in neighbors_dict.items()}

    for a in range(len(original_molecule)):
        set_a = neighbor_sets[a]
        for n in neighbors_dict[a]:
            if n <= a:
                continue                          # process each edge (a,n) with a<n once
            for n2 in set_a & neighbor_sets[n]:   # common neighbours of a and n
                if n2 <= n:
                    continue                      # ensures a < n < n2 → unique triangle
                mid_vect = (all_positions[a] + all_positions[n] + all_positions[n2]) / 3
                xxx_mol.append(Atom(symbol=specie, position=mid_vect))

    return xxx_mol

#------------------------------------------------------------------------------------------
def add_ineq_interstitial_atoms_with_expansion(original_molecule, atom_list):
    """Build a list of grown clusters by adding each candidate atom after a focused expansion.

    in:
        original_molecule (ase.Atoms): base cluster structure before growing.
        atom_list (list of ase.Atom): candidate atoms with their positions to add one at a time.
    out:
        list of ase.Atoms: one extended cluster per candidate atom, each with a focused
            expansion applied before the atom is appended.
    """
    molist_out = []
    for add_atom in atom_list:
        mid_vect = np.array(add_atom.position)
        add_mol = original_molecule.copy()
        exp_mol = focused_expansion(add_mol, mid_vect)
        exp_mol.append(add_atom)
        molist_out.append(exp_mol)
    return molist_out

#------------------------------------------------------------------------------------------
def process_single_molecule(args):
    """Process one cluster molecule: find triangles, add interstitial atoms, and expand.

    in:
        args (tuple): (imol, specie, dtol) where imol is an ase.Atoms cluster, specie is
            the chemical symbol of the new atom, and dtol is the bond-detection scale factor.
    out:
        list of ase.Atoms: all grown structures derived from imol by adding one
            symmetry-inequivalent interstitial atom at a time.
    """
    imol, specie, dtol = args
    org_nnn = len(imol)
    org_mol = imol.copy()
    org_mol.info['e'] = 0.0
    org_mol.translate(-org_mol.get_center_of_mass())

    all_neighbors = neighbor_finder(org_mol, dtol)
    xxx_mol = add_all_interstitial_atoms(org_mol, all_neighbors, specie)
    inequivalent, _ = get_inequivalent(xxx_mol)
    #inequivalent = get_inequivalent(xxx_mol)
    elegible_atoms = [xxx_mol[i] for i in inequivalent if i >= org_nnn]
    mod_mol = add_ineq_interstitial_atoms_with_expansion(org_mol, elegible_atoms)
    mod_mol = rename(mod_mol, imol.info['i'], 4)

    return mod_mol

#------------------------------------------------------------------------------------------
def growpal_parallel(poscarlist, specie, dtol=1.2, n_cores=None):
    """Grow all clusters in a list by one atom using multiprocessing (GrowPal algorithm).

    in:
        poscarlist (list of ase.Atoms): input clusters of size N to grow to size N+1.
        specie (str): chemical symbol of the atom to add.
        dtol (float): bond-detection scale factor for adjacency (default 1.2).
        n_cores (int or None): number of parallel worker processes; None uses all available
            cores minus one.
    out:
        list of ase.Atoms: all candidate clusters of size N+1, collected from all workers.
    """
    start = time.time()

    if n_cores is None:
        n_cores = max(1, cpu_count() - 1)

    args_list = [(imol, specie, dtol) for imol in poscarlist]

    molist_out = []
    with Pool(processes=n_cores) as pool:
        for result in pool.imap_unordered(process_single_molecule, args_list):
            molist_out.extend(result)

    end = time.time()
    print('GrowPal generation at %5.2f s' % (end - start))

    return molist_out

#------------------------------------------------------------------------------------------
def display_info(moleculein, stage_string):
    """Print a ranked energy summary table for a list of clusters.

    in:
        moleculein (list of ase.Atoms): cluster structures; each must have info['e'],
            info['i'], and info['c'] (convergence flag).
        stage_string (str): label shown in the summary header.
    out:
        None.
    """
    print('%s SUMMARY' % (stage_string))
    print("Number File--------Name   Energy (ev)   Delta----E T")
    molzz = sort_by_energy(moleculein, 1)
    emin = molzz[0].info['e']
    for ii, imol in enumerate(molzz):
        ei = imol.info['e']
        id = imol.info['i']
        nt = imol.info['c']
        deltae  =  ei - emin
        kk=str(ii+1).zfill(6)
        print("%s %s %13.8f %12.8f %d" %(kk, id, ei, deltae, nt))
