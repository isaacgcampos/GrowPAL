import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import connected_components
#----------------------------------------------------------------------------------------------------------
def _make_connected(connectivity, n_samples):
    """Add minimal bridging edges so the connectivity graph has exactly one component."""
    n_comp, labels = connected_components(connectivity, directed=False)
    if n_comp == 1:
        return connectivity
    conn = connectivity.tolil()
    representatives = [int(np.where(labels == i)[0][0]) for i in range(n_comp)]
    for k in range(n_comp - 1):
        a, b = representatives[k], representatives[k + 1]
        conn[a, b] = 1.0
        conn[b, a] = 1.0
    return conn.tocsr()
#----------------------------------------------------------------------------------------------------------
def clustering_agg(lista, descriptors, n_clusters, n_neighbors=15, use_connectivity=True):
    """Cluster a list of structures using agglomerative (Ward) clustering on precomputed descriptors.

    When use_connectivity=True a k-NN connectivity graph is used so complexity drops
    from O(N^2) to O(N*n_neighbors), making it practical for N in the tens of thousands.
    When use_connectivity=False Ward operates on the full distance matrix (O(N^2)) which
    produces more globally coherent partitions at the cost of higher memory usage.

    in:
        lista (list of ase.Atoms): structures to cluster.
        descriptors (list of numpy.ndarray): one descriptor vector per structure.
        n_clusters (int): number of clusters to form.
        n_neighbors (int): neighbors for the k-NN connectivity graph (default 15);
                           only used when use_connectivity=True.
        use_connectivity (bool): if True use a kNN graph constraint (default True);
                                 if False use full Ward without connectivity.
    out:
        dict: mapping of cluster label (int) to list of ase.Atoms structures in that
            cluster.
    """
    if len(descriptors) < n_clusters:
        print("La cantidad de descriptores es menor que el número de clusters. Devolviendo la lista original.")
        return {0: lista}

    if use_connectivity:
        d = descriptors.shape[1]
        n_floor = max(n_neighbors, int(2 * d**0.5))
        n_eff = min(max(n_floor, len(descriptors) // 500), len(descriptors) - 1)
        connectivity = kneighbors_graph(descriptors, n_neighbors=n_eff,
                                        include_self=False, n_jobs=-1)
        connectivity = (connectivity + connectivity.T) / 2
        connectivity = _make_connected(connectivity, len(descriptors))
        model = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity, linkage='ward')
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(descriptors)

    clusters = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(labels):
        clusters[label].append(lista[idx])

    return clusters
