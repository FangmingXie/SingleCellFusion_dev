#!/usr/bin/env python3
"""
louvain clustering from kNN graph
2 model parameters: # of PCs and # of nearest neighbors

updated from CEMBA_clustering_louvain_jaccard.py for large number of cells
"""
from __init__ import *
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import louvain
import igraph as ig
from scipy import sparse
from annoy import AnnoyIndex

from snmcseq_utils import create_logger

# major change in annoy functions 5/7/2019 
def build_knn_map(X, metric='euclidean', n_trees=10, verbose=True):
    """X is expected to have low feature dimensions (n_obs, n_features) with (n_features <= 50)

    return:
         t: annoy knn object, can be used in the following ways 
                t.get_nns_by_vector
                t.get_nns_by_item
    """
    ti = time.time()

    n_obs, n_f = X.shape
    t = AnnoyIndex(n_f, metric=metric)  # Length of item vector that will be indexed
    for i, X_row in enumerate(X):
        t.add_item(i, X_row)
    t.build(n_trees) # 10 trees
    if verbose:
        print("Time used to build kNN map {}".format(time.time()-ti))
    return t 

def get_knn_by_items(t, k, 
    form='list', 
    search_k=-1, 
    include_distances=False,
    verbose=True, 
    ):
    """Get kNN for each item in the knn map t
    """
    ti = time.time()
    # set up
    n_obs = t.get_n_items()
    n_f = t.f
    if k > n_obs:
        print("Actual k: {}->{} due to low n_obs".format(k, n_obs))
        k = n_obs

    knn = [0]*(n_obs)
    knn_dist = [0]*(n_obs)
    # this block of code can be optimized
    if include_distances:
        for i in range(n_obs):
            res = t.get_nns_by_item(i, k, search_k=search_k, include_distances=include_distances)
            knn[i] = res[0]
            knn_dist[i] = res[1]
    else:
        for i in range(n_obs):
            res = t.get_nns_by_item(i, k, search_k=search_k, include_distances=include_distances) 
            knn[i] = res

    knn = np.array(knn)
    knn_dist = np.array(knn_dist)

    if verbose:
        print("Time used to get kNN {}".format(time.time()-ti))

    if form == 'adj':
        # row col 1/dist 
        row_inds = np.repeat(np.arange(n_obs), k)
        col_inds = np.ravel(knn)
        if include_distances:
            data = np.ravel(knn_dist) 
        else:
            data = [1]*len(row_inds)
        knn_dist_mat = sparse.coo_matrix((data, (row_inds, col_inds)), shape=(n_obs, n_obs))
        return knn_dist_mat
    elif form == 'list':  #
        if include_distances:
            return knn, knn_dist
        else:
            return knn
    else:
        raise ValueError("Choose from 'adj' and 'list'")

def get_knn_by_vectors(t, X, k, 
    form='list', 
    search_k=-1, 
    include_distances=False,
    verbose=True, 
    ):
    """Get kNN for each row vector of X 
    """
    ti = time.time()
    # set up
    n_obs = t.get_n_items()
    n_f = t.f
    n_obs_test, n_f_test = X.shape
    assert n_f_test == n_f

    if k > n_obs:
        print("Actual k: {}->{} due to low n_obs".format(k, n_obs))
        k = n_obs

    knn = [0]*(n_obs_test)
    knn_dist = [0]*(n_obs_test)
    if include_distances:
        for i, vector in enumerate(X):
            res = t.get_nns_by_vector(vector, k, search_k=search_k, include_distances=include_distances) 
            knn[i] = res[0]
            knn_dist[i] = res[1]
    else:
        for i, vector in enumerate(X):
            res = t.get_nns_by_vector(vector, k, search_k=search_k, include_distances=include_distances) 
            knn[i] = res

    knn = np.array(knn)
    knn_dist = np.array(knn_dist)

    if verbose:
        print("Time used to get kNN {}".format(time.time()-ti))

    if form == 'adj':
        # row col 1/dist 
        row_inds = np.repeat(np.arange(n_obs_test), k)
        col_inds = np.ravel(knn)
        if include_distances:
            data = np.ravel(knn_dist) 
        else:
            data = [1]*len(row_inds)
        knn_dist_mat = sparse.coo_matrix((data, (row_inds, col_inds)), shape=(n_obs_test, n_obs))
        return knn_dist_mat
    elif form == 'list':  #
        if include_distances:
            return knn, knn_dist
        else:
            return knn
    else:
        raise ValueError("Choose from 'adj' and 'list'")

def gen_knn_annoy(X, k, form='list', 
    metric='euclidean', n_trees=10, search_k=-1, verbose=True, 
    include_distances=False,
    ):
    """X is expected to have low feature dimensions (n_obs, n_features) with (n_features <= 50)
    """
    ti = time.time()

    n_obs, n_f = X.shape
    t = build_knn_map(X, metric=metric, n_trees=n_trees, verbose=verbose)

    return get_knn_by_items(t, k,                             
                            form=form, 
                            search_k=search_k, 
                            include_distances=include_distances,
                            verbose=verbose, 
                            )

def gen_knn_annoy_train_test(X_train, X_test, k, 
    form='list', 
    metric='euclidean', n_trees=10, search_k=-1, verbose=True, 
    include_distances=False,
    ):
    """X is expected to have low feature dimensions (n_obs, n_features) with (n_features <= 50)
    For each row in X_test, find k nearest neighbors in X_train
    """
    ti = time.time()
    
    n_obs, n_f = X_train.shape
    n_obs_test, n_f_test = X_test.shape
    assert n_f == n_f_test 
    
    t = build_knn_map(X_train, metric=metric, n_trees=n_trees, verbose=verbose)
    return get_knn_by_vectors(t, X_test, k, 
                                form=form, 
                                search_k=search_k, 
                                include_distances=include_distances,
                                verbose=verbose, 
                                )

def gen_knn(pcX, k, form='adj', metric='euclidean', verbose=True): 
    """Generate kNN matrix from a pcX (n_obs, n_feature) matrix
    """
    ti = time.time()

    n_obs, n_f = X.shape
    k = min(k, n_obs)
    
    knn = NearestNeighbors(n_neighbors=k, metric=metric).fit(pcX)
    
    if form == 'adj':
        g_knn = knn.kneighbors_graph(pcX, mode='connectivity')
        if verbose:
            print("Time spent on generate kNN graph: {}".format(time.time()-ti))
        return g_knn
            
    elif form == 'list':
        dists, inds = knn.kneighbors(pcX)
        if verbose:
            print("Time spent on generate kNN graph: {}".format(time.time()-ti))
        return (dists, inds) 
    
def compute_jaccard_weights_from_knn(X):
    """compute jaccard index on a knn graph
    Arguments: 
        X (unweighted) kNN ajacency matrix (each row Xi* gives the kNNs of cell i) 
        X has to be 0-1 valued 
        k (number of nearest neighbors) 
        
    output: numpy matrix Y
    """
    X = sparse.csr_matrix(X)
    ni, nj = X.shape
    assert ni == nj
    
    k = X[0, :].sum() # number of neighbors
    
    Y = X.dot(X.T)
    # Y = X.multiply(tmp/(2*k - tmp.todense()))    
    Y.data = Y.data/(2*k - Y.data)
    
    return Y 

def adjacency_to_igraph(adj_mtx, weighted=False):
    """
    Converts an adjacency matrix to an igraph object
    
    Args:
        adj_mtx (sparse matrix): Adjacency matrix
        directed (bool): If graph should be directed
    
    Returns:
        G (igraph object): igraph object of adjacency matrix
    
    Uses code from:
        https://github.com/igraph/python-igraph/issues/168
        https://stackoverflow.com/questions/29655111

    Author:
        Wayne Doyle 
        (Fangming Xie modified) 
    """
    nrow, ncol = adj_mtx.shape
    if nrow != ncol:
        raise ValueError('Adjacency matrix should be a square matrix')
    vcount = nrow
    sources, targets = adj_mtx.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))
    G = ig.Graph(n=vcount, edges=edgelist, directed=True)
    if weighted:
        G.es['weight'] = adj_mtx.data
    return G

def adjacency_to_nxgraph(adj_mtx, weighted=False):
    """
    Converts an adjacency matrix to an networkx object
    Args:
        adj_mtx (sparse matrix): Adjacency matrix
    Returns:
        G (networkx object): networkx object of adjacency matrix
    """
    import networkx as nx

    nrow, ncol = adj_mtx.shape
    if nrow != ncol:
        raise ValueError('Adjacency matrix should be a square matrix')
    sources, targets = adj_mtx.nonzero()
    G = nx.Graph()
    if not weighted:
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        G.add_edges_from(edgelist) # non-direct graph only
    else:
        edgelist = list(zip(sources.tolist(), targets.tolist(), weights.tolist()))
        G.add_weighted_edges_from(edgelist) # non-direct graph only
    return G

def louvain_lite(G, cell_list, weighted=False, verbose=True):
    """weighted=False is 10x faster than True
    """
    ti = time.time()
        
    if weighted:
        partition1 = louvain.find_partition(G, louvain.ModularityVertexPartition, 
                                            weights=G.es["weight"]
                                           )
    else:
        partition1 = louvain.find_partition(G, louvain.ModularityVertexPartition) 
        
    labels = [0]*(len(cell_list)) 
    for i, cluster in enumerate(partition1):
        for element in cluster:
            labels[element] = i+1

    df_res = pd.DataFrame(index=cell_list)
    df_res['cluster'] = labels 
    df_res = df_res.rename_axis('sample', inplace=False)
    
    if verbose:
        print("Time spent on louvain clustering: {}".format(time.time()-ti))
        
    return df_res


def leiden_lite(g, cell_list, resolution=1, weighted=False, verbose=True, num_starts=None, seed=1):
    """ Code from Ethan Armand and Wayne Doyle, ./mukamel_lab/mop
    slightly modified by Fangming Xie 05/13/2019
    """
    import leidenalg
    
    ti = time.time()
    
    if num_starts is not None:
        np.random.seed(seed)
        partitions = []
        quality = []
        seeds = np.random.randint(10*num_starts, size=num_starts)
        for seed in seeds:
            if weighted:
                temp_partition = leidenalg.find_partition(g,
                                                      leidenalg.RBConfigurationVertexPartition, 
                                                      weights=g.es['weight'],
                                                      resolution_parameter=resolution,
                                                      seed=seed,
                                                      )
            else:
                temp_partition = leidenalg.find_partition(g,
                                                      leidenalg.RBConfigurationVertexPartition,
                                                      resolution_parameter=resolution,
                                                      seed=seed,
                                                      )
            quality.append(temp_partition.quality())
            partitions.append(temp_partition)
        partition1 = partitions[np.argmax(quality)]
    else:
        if weighted:
            partition1 = leidenalg.find_partition(g,
                                                  leidenalg.RBConfigurationVertexPartition,
                                                  weights=g.es['weight'],
                                                  resolution_parameter=resolution,
                                                  seed=seed,
                                                  )
        else:
            partition1 = leidenalg.find_partition(g,
                                                  leidenalg.RBConfigurationVertexPartition,
                                                  resolution_parameter=resolution,
                                                  seed=seed,
                                                  )

    # get cluster labels from partition1
    labels = [0]*(len(cell_list)) 
    for i, cluster in enumerate(partition1):
        for element in cluster:
            labels[element] = i+1

    df_res = pd.DataFrame(index=cell_list)
    df_res['cluster'] = labels 
    df_res = df_res.rename_axis('sample', inplace=False)
    
    if verbose:
        print("Time spent on leiden clustering: {}".format(time.time()-ti))
        
    return df_res

def clustering_routine(X, cell_list, k, 
    seed=1, verbose=True,
    resolution=1, metric='euclidean', option='plain', n_trees=10, search_k=-1, num_starts=None):
    """
    X is a (n_obs, n_feature) matrix, n_feature <=50 is recommended
    option: {'plain', 'jaccard', ...}
    """
    assert len(cell_list) == len(X)
    
    if option == 'plain':
        g_knn = gen_knn_annoy(X, k, form='adj', metric=metric, 
                              n_trees=n_trees, search_k=search_k, verbose=verbose)
        G = adjacency_to_igraph(g_knn, weighted=False)
        # df_res = louvain_lite(G, cell_list, weighted=False)
        df_res = leiden_lite(G, cell_list, resolution=resolution, seed=seed, 
                            weighted=False, verbose=verbose, num_starts=num_starts)
        
    elif option == 'jaccard':
        g_knn = gen_knn_annoy(X, k, form='adj', metric=metric, 
                              n_trees=n_trees, search_k=search_k, verbose=verbose)
        gw_knn = compute_jaccard_weights_from_knn(g_knn)
        G = adjacency_to_igraph(gw_knn, weighted=True)
        # df_res = louvain_lite(G, cell_list, weighted=True)
        df_res = leiden_lite(G, cell_list, resolution=resolution, seed=seed, 
                            weighted=True, verbose=verbose, num_starts=num_starts)
    else:
        raise ValueError('Choose from "plain" and "jaccard"')
    
    return df_res

def clustering_routine_multiple_resolutions(X, cell_list, k, 
    seed=1, verbose=True,
    resolutions=[1], metric='euclidean', option='plain', n_trees=10, search_k=-1, num_starts=None):
    """
    X is a (n_obs, n_feature) matrix, n_feature <=50 is recommended
    option: {'plain', 'jaccard', ...}
    """
    assert len(cell_list) == len(X)
    
    res = []
    if option == 'plain':
        g_knn = gen_knn_annoy(X, k, form='adj', metric=metric, 
                              n_trees=n_trees, search_k=search_k, verbose=verbose)
        G = adjacency_to_igraph(g_knn, weighted=False)
        for resolution in resolutions:
            df_res = leiden_lite(G, cell_list, resolution=resolution, seed=seed, 
                                weighted=False, verbose=verbose, num_starts=num_starts)
            df_res = df_res.rename(columns={'cluster': 'cluster_r{}'.format(resolution)})
            res.append(df_res)
        
    elif option == 'jaccard':
        g_knn = gen_knn_annoy(X, k, form='adj', metric=metric, 
                              n_trees=n_trees, search_k=search_k, verbose=verbose)
        gw_knn = compute_jaccard_weights_from_knn(g_knn)
        G = adjacency_to_igraph(gw_knn, weighted=True)
        for resolution in resolutions:
            df_res = leiden_lite(G, cell_list, resolution=resolution, seed=seed, 
                                weighted=True, verbose=verbose, num_starts=num_starts)
            df_res = df_res.rename(columns={'cluster': 'cluster_r{}'.format(resolution)})
            res.append(df_res)
        
    else:
        raise ValueError('Choose from "plain" and "jaccard"')
    res = pd.concat(res, axis=1)
    
    return res


# def louvain_lite_resolution(G, cell_list, resolution=1, verbose=True):
#     """
#     weighted=False is 10x faster than True
#     """
#     import community

#     ti = time.time()
        
#     partition1 = community.best_partition(G, resolution=resolution, randomize=True)

#     cell_idx = []
#     labels = []
#     for key, val in partition1.items():
#         cell_idx.append(key)
#         labels.append(val)

#     df_res = pd.DataFrame(index=np.array(cell_list)[cell_idx])
#     df_res['cluster'] = labels 
#     df_res = df_res.rename_axis('sample', inplace=False)
    
#     if verbose:
#         print("Time spent on louvain clustering: {}".format(time.time()-ti))
#     return df_res


# def clustering_routine_old(X, cell_list, k, metric='euclidean', option='plain'):
#     """
#     X is a (n_obs, n_feature) matrix, n_feature <=50 is recommended
#     option: {'plain', 'jaccard', ...}
#     """
#     if option == 'plain':
#         g_knn = gen_knn(X, k, form='adj', metric=metric, verbose=True)
#         G = adjacency_to_igraph(g_knn, weighted=False)
#         df_res = louvain_lite(G, cell_list, weighted=False)
        
#     elif option == 'jaccard':
#         g_knn = gen_knn(X, k, form='adj', metric=metric, verbose=True)
#         gw_knn = compute_jaccard_weights_from_knn(g_knn)
#         G = adjacency_to_igraph(gw_knn, weighted=True)
#         df_res = louvain_lite(G, cell_list, weighted=True)
#     else:
#         raise ValueError('Choose from "plain" and "jaccard"')
    
#     return df_res

# def clustering_routine_resolution(X, cell_list, k, resolution=1, 
#     metric='euclidean', option='plain', n_trees=10, search_k=-1):
#     """Non-directed graph only
#     X is a (n_obs, n_feature) matrix, n_feature <=50 is recommended
#     option: {'plain', 'jaccard', ...}
#     """
#     assert option in ['plain', 'jaccard']

#     if option == 'plain':
#         g_knn = gen_knn_annoy(X, k, form='adj', metric=metric, 
#                               n_trees=n_trees, search_k=search_k, verbose=True)
#         G = adjacency_to_nxgraph(g_knn, weighted=False)
#         df_res = louvain_lite_resolution(G, cell_list, resolution=resolution, verbose=True)
        
#     elif option == 'jaccard':
#         g_knn = gen_knn_annoy(X, k, form='adj', metric=metric, 
#                               n_trees=n_trees, search_k=search_k, verbose=True)
#         gw_knn = compute_jaccard_weights_from_knn(g_knn)
#         G = adjacency_to_nxgraph(gw_knn, weighted=True)
#         df_res = louvain_lite_resolution(G, cell_list, resolution=resolution, verbose=True)
        
#     else:
#         raise ValueError('Choose from "plain" and "jaccard"')
    
#     return df_res



