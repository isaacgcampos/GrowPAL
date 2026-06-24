import time
from joblib import Parallel, delayed
from growpal.libclustering  import clustering_agg
from growpal.libdescriptors import mbtr_comb_fast, mbtr_comb_dscribe
from aegon.libutils         import sort_by_energy

def select_by_clustering(mol_in, selection_count=160, n_clusters=8, n_jobs=1, special_mols=None, use_connectivity=True, mono=True):
    """Select a subset of structures using MBTR descriptors and Ward clustering.

    Computes combined MBTR_dis + MBTR_cos descriptors, clusters with
    AgglomerativeClustering (Ward), and takes the selection_count
    lowest-energy structures from each cluster.

    When special_mols is provided those structures bypass clustering and are
    always included verbatim in the output.

    in:
        mol_in          (list of ase.Atoms): input cluster structures.
        selection_count (int)              : number of structures to select per cluster (default 160).
        n_clusters      (int)              : total number of clusters including the special one
                                             (default 8); regular clustering uses n_clusters-1.
        n_jobs          (int)              : parallel workers for descriptor computation and
                                             intra-cluster selection; 1 = serial (default).
        special_mols    (list or None)     : structures that bypass clustering and are
                                             always included in the output (default None).
        use_connectivity (bool)            : if True use kNN graph constraint in Ward
                                             clustering (default True); if False use full
                                             Ward without connectivity (O(N^2), more
                                             globally coherent partitions).
        mono            (bool)             : if True use the numba fast path (monoatomic,
                                             no dscribe); if False use dscribe MBTR for
                                             multi-species systems (default True).
    out:
        list of ase.Atoms: selected structures sorted by energy.
    """
    start = time.time()

    # --- split special / regular ---
    if special_mols:
        special_ids  = {id(m) for m in special_mols}
        mol_regular  = [m for m in mol_in if id(m) not in special_ids]
        n_reg        = n_clusters - 1
        print('Special cluster  : %3d members  |  regular pool: %d  |  quota/cluster: %d'
              % (len(special_mols), len(mol_regular), selection_count))
    else:
        mol_regular  = mol_in
        n_reg        = n_clusters

    nn        = len(mol_regular)
    total_reg = selection_count * n_reg

    if total_reg <= 0 or n_reg <= 0:
        return sort_by_energy(special_mols or [], 1)
    if nn <= total_reg:
        return sort_by_energy(mol_regular + (special_mols or []), 1)

    # --- descriptors ---
    t0  = time.time()
    if mono:
        des = mbtr_comb_fast(mol_regular)                # (N, 400) ??? numba, no dscribe
    else:
        des = mbtr_comb_dscribe(mol_regular, n_jobs)     # (N, F)   ??? dscribe, multi-species
    t1  = time.time()
    print('Descriptors      at %6.2f s [%d structs, shape %s]'
          % (t1 - t0, nn, des.shape))

    # --- AgglomerativeClustering (Ward, with or without k-NN connectivity) ---
    t2       = time.time()
    clusters = clustering_agg(mol_regular, des, n_reg, use_connectivity=use_connectivity)
    t3       = time.time()
    clus_label = 'Ward/kNN' if use_connectivity else 'Ward/full'
    print('Clustering (%s) at %6.2f s [%d clusters]' % (clus_label, t3 - t2, n_reg))

    t4         = time.time()
    del des
    dyn_counts = {k: selection_count for k in clusters}

    def _select_one(clave, valor):
        count = dyn_counts.get(clave, 1)
        taken = sorted(valor, key=lambda a: a.info['e'])[:count]
        return clave, len(valor), count, taken

    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_select_one)(clave, valor) for clave, valor in clusters.items()
    )
    lista_est = []
    for clave, sz, count, taken in results:
        lista_est.extend(taken)
        print('  cluster %d: size=%6d  quota=%4d  taken=%4d' % (clave+1, sz, count, len(taken)))
    if special_mols:
        lista_est.extend(special_mols)
    mol_out = sort_by_energy(lista_est, 1)
    del lista_est
    nf      = len(mol_out)
    end     = time.time()
    print('Selection        at %6.2f s [%d -> %d]' % (end - t4, nn, nf))
    return mol_out
