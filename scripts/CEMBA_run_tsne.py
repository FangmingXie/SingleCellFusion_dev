"""Generate tSNE coordinates
"""
from __init__ import *
from snmcseq_utils import create_logger

def run_tsne(df, perp=30, n_pc=50, n_tsne=2, 
             random_state=1, output_file=None, sample_column_suffix=None, **kwargs):
    """run tsne on "_mcc$" columns
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE 

    ti = time.time()

    if sample_column_suffix: 
        df = df.filter(regex='{}$'.format(sample_column_suffix))
    logging.info("Running tsne: {} PC, {} perp, {} dim.\nInput shape: {}".format(n_pc, perp, n_tsne, df.shape))
    
    pca = PCA(n_components=n_pc)
    pcs = pca.fit_transform(df.T)

    tsne = TSNE(n_components=n_tsne, init='pca', random_state=random_state, perplexity=perp, **kwargs)
    ts = tsne.fit_transform(pcs)
 
    if n_tsne == 2: 
        df_tsne = pd.DataFrame(ts, columns=['tsne_x','tsne_y'])
    elif n_tsne == 3:
        df_tsne = pd.DataFrame(ts, columns=['tsne_x','tsne_y', 'tsne_z'])

    if sample_column_suffix:
        df_tsne['sample'] = [sample[:-len(sample_column_suffix)] for sample in df.columns.tolist()]
    else:
        df_tsne['sample'] = df.columns.tolist()
    df_tsne = df_tsne.set_index('sample')
    
    if output_file:
        df_tsne.to_csv(output_file, sep="\t", na_rep='NA', header=True, index=True)
        logging.info("Saved tsne coordinates to file. {}".format(output_file))

    tf = time.time()
    logging.info("Done with tSNE. running time: {} seconds.".format(tf - ti))
    
    return df_tsne

def run_umap_lite(X, cell_list, n_neighbors=15, min_dist=0.1, n_dim=2, 
             random_state=1, output_file=None, **kwargs):
    """run umap on X (n_obs, n_features) 
    """
    from sklearn.decomposition import PCA
    from umap import UMAP
    # from sklearn.manifold import TSNE 

    ti = time.time()

    logging.info("Running UMAP: {} n_neighbors, {} min_dist , {} dim.\nInput shape: {}"
                        .format(n_neighbors, min_dist, n_dim, X.shape))
    
    umap = UMAP(n_components=n_dim, random_state=random_state, 
                n_neighbors=n_neighbors, min_dist=min_dist, **kwargs)
    ts = umap.fit_transform(X)
 
    if n_dim == 2: 
        df_tsne = pd.DataFrame(ts, columns=['tsne_x','tsne_y'])
    elif n_dim == 3:
        df_tsne = pd.DataFrame(ts, columns=['tsne_x','tsne_y', 'tsne_z'])

    df_tsne['sample'] = cell_list 
    df_tsne = df_tsne.set_index('sample')
    
    if output_file:
        df_tsne.to_csv(output_file, sep="\t", na_rep='NA', header=True, index=True)
        logging.info("Saved tsne coordinates to file. {}".format(output_file))

    tf = time.time()
    logging.info("Done with tSNE. running time: {} seconds.".format(tf - ti))
    
    return df_tsne


def run_umap(df, n_neighbors=15, min_dist=0.1, n_pc=50, n_dim=2, 
             random_state=1, output_file=None, sample_column_suffix=None, **kwargs):
    """run tsne on "_mcc$" columns
    """
    from sklearn.decomposition import PCA
    from umap import UMAP
    # from sklearn.manifold import TSNE 

    ti = time.time()

    if sample_column_suffix: 
        df = df.filter(regex='{}$'.format(sample_column_suffix))
    logging.info("Running tsne: {} PC, {} n_neighbors, {} min_dist , {} dim.\nInput shape: {}".format(n_pc, n_neighbors, min_dist, n_dim, df.shape))
    
    pca = PCA(n_components=n_pc)
    pcs = pca.fit_transform(df.T)

    # tsne = TSNE(n_components=n_tsne, init='pca', random_state=random_state, perplexity=perp, **kwargs)
    # ts = tsne.fit_transform(pcs)
    umap = UMAP(n_components=n_dim, random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist, **kwargs)
    ts = umap.fit_transform(pcs)
 
    if n_dim == 2: 
        df_tsne = pd.DataFrame(ts, columns=['tsne_x','tsne_y'])
    elif n_dim == 3:
        df_tsne = pd.DataFrame(ts, columns=['tsne_x','tsne_y', 'tsne_z'])

    if sample_column_suffix:
        df_tsne['sample'] = [sample[:-len(sample_column_suffix)] for sample in df.columns.tolist()]
    else:
        df_tsne['sample'] = df.columns.tolist()
    df_tsne = df_tsne.set_index('sample')
    
    if output_file:
        df_tsne.to_csv(output_file, sep="\t", na_rep='NA', header=True, index=True)
        logging.info("Saved tsne coordinates to file. {}".format(output_file))

    tf = time.time()
    logging.info("Done with tSNE. running time: {} seconds.".format(tf - ti))
    
    return df_tsne


def run_tsne_v2(df, perp=30, n_pc=50, n_tsne=2, 
             random_state=1, output_file=None, sample_column_suffix=None, nthreads=1, **kwargs):
    """run tsne on "_mcc$" columns
    """
    from sklearn.decomposition import PCA
    # from sklearn.manifold import TSNE 
    import fitsne


    ti = time.time()

    if sample_column_suffix: 
        df = df.filter(regex='{}$'.format(sample_column_suffix))
    logging.info("Running tsne: {} PC, {} perp, {} dim.\nInput shape: {}".format(n_pc, perp, n_tsne, df.shape))
    
    pca = PCA(n_components=n_pc)
    pcs = pca.fit_transform(df.T)

    # tsne = TSNE(n_components=n_tsne, init='pca', random_state=random_state, perplexity=perp, **kwargs)
    #   ts = tsne.fit_transform(pcs)
    pcs = pcs.copy(order='C')
    ts = fitsne.FItSNE(pcs, perplexity=perp, rand_seed=random_state, nthreads=nthreads, **kwargs)
 
    if n_tsne == 2: 
        df_tsne = pd.DataFrame(ts, columns=['tsne_x','tsne_y'])
    elif n_tsne == 3:
        df_tsne = pd.DataFrame(ts, columns=['tsne_x','tsne_y', 'tsne_z'])

    if sample_column_suffix:
        df_tsne['sample'] = [sample[:-len(sample_column_suffix)] for sample in df.columns.tolist()]
    else:
        df_tsne['sample'] = df.columns.tolist()
    df_tsne = df_tsne.set_index('sample')
    
    if output_file:
        df_tsne.to_csv(output_file, sep="\t", na_rep='NA', header=True, index=True)
        logging.info("Saved tsne coordinates to file. {}".format(output_file))

    tf = time.time()
    logging.info("Done with tSNE. running time: {} seconds.".format(tf - ti))
    
    return df_tsne
    
