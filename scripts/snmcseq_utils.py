"""
Fangming Xie, Chris Keown
Many functions are adopted from Chris's mypy
"""

from __init__ import *

import subprocess as sp
import os
from scipy import sparse

def diag_matrix_old(X, rows=np.array([]), cols=np.array([])):
    """Diagonalize a matrix as much as possible
    """
    di, dj = X.shape
    new_X = X.copy()
    new_rows = rows.copy() 
    new_cols = cols.copy() 
    
    if new_rows.size == 0:
        new_rows = np.arange(di)
    if new_cols.size == 0:
        new_cols = np.arange(dj)
        
    # 
    for idx in range(min(di, dj)):
        T = new_X[idx: , idx: ]
        i, j = np.unravel_index(T.argmax(), T.shape)

        # swap row idx, idx+i
        tmp = new_X[idx, :].copy()
        new_X[idx, :] = new_X[idx+i, :].copy() 
        new_X[idx+i, :] = tmp 
        
        tmp = new_rows[idx]
        new_rows[idx] = new_rows[idx+i]
        new_rows[idx+i] = tmp

        # swap col idx, idx+j
        tmp = new_X[:, idx].copy()
        new_X[:, idx] = new_X[:, idx+j].copy() 
        new_X[:, idx+j] = tmp 
        
        tmp = new_cols[idx]
        new_cols[idx] = new_cols[idx+j]
        new_cols[idx+j] = tmp
    # 
    if di == dj:
        pass
    elif di < dj:
        part_mat =  new_X[:, di:].copy()
        new_ids = (part_mat.max(axis=0).argsort()[::-1])
        tmp = part_mat[:, new_ids].copy()
        new_X[:, di:] = tmp 
        
        new_cols[di:] = new_cols[di+new_ids].copy()
        
    elif di > dj:
        part_mat =  new_X[dj:, :].copy()
        new_ids = (part_mat.max(axis=1).argsort()[::-1])
        tmp = part_mat[new_ids, :].copy()
        new_X[dj:, :] = tmp 
        
        new_rows[dj:] = new_rows[dj+new_ids].copy()
        
    return new_X, new_rows, new_cols 

def diag_matrix(X, rows=np.array([]), cols=np.array([]), threshold=None):
    """Diagonalize a matrix as much as possible
    """
    di, dj = X.shape
    transposed = 0
    
    if di > dj:
        di, dj = dj, di
        X = X.T.copy()
        rows, cols = cols.copy(), rows.copy()
        transposed = 1
        
    # start (di <= dj)
    new_X = X.copy()
    new_rows = rows.copy() 
    new_cols = cols.copy() 
    if new_rows.size == 0:
        new_rows = np.arange(di)
    if new_cols.size == 0:
        new_cols = np.arange(dj)
        
    # bring the greatest values in the lower right matrix to diagnal position 
    for idx in range(min(di, dj)):

        T = new_X[idx: , idx: ]
        i, j = np.unravel_index(T.argmax(), T.shape) # get the coords of the max element of T
        
        if threshold and T[i, j] < threshold:
            dm = idx # new_X[:dm, :dm] is done (0, 1, ..., dm-1) excluding dm
            break
        else:
            dm = idx+1 # new_X[:dm, :dm] will be done

        # swap row idx, idx+i
        tmp = new_X[idx, :].copy()
        new_X[idx, :] = new_X[idx+i, :].copy() 
        new_X[idx+i, :] = tmp 
        
        tmp = new_rows[idx]
        new_rows[idx] = new_rows[idx+i]
        new_rows[idx+i] = tmp

        # swap col idx, idx+j
        tmp = new_X[:, idx].copy()
        new_X[:, idx] = new_X[:, idx+j].copy() 
        new_X[:, idx+j] = tmp 
        
        tmp = new_cols[idx]
        new_cols[idx] = new_cols[idx+j]
        new_cols[idx+j] = tmp
        
    # 
    if dm == dj:
        pass
    elif dm < dj: # free columns

        col_dict = {}
        sorted_col_idx = np.arange(dm)
        free_col_idx = np.arange(dm, dj)
        linked_rowcol_idx = new_X[:, dm:].argmax(axis=0)
        
        for col in sorted_col_idx:
            col_dict[col] = [col]
        for col, key in zip(free_col_idx, linked_rowcol_idx): 
            if key < dm:
                col_dict[key] = col_dict[key] + [col]
            else:
                col_dict[key] = [col]
                
            
        new_col_order = np.hstack([col_dict[key] for key in sorted(col_dict.keys())])
        
        # update new_X new_cols
        new_X = new_X[:, new_col_order].copy()
        new_cols = new_cols[new_col_order]
    else:
        raise ValueError("Unexpected situation: dm > dj")
    
    if transposed:
        new_X = new_X.T
        new_rows, new_cols = new_cols, new_rows
    return new_X, new_rows, new_cols 

def diag_matrix_rows(X, rows=np.array([]), cols=np.array([]),):
    """Diagonalize a matrix as much as possible by only rearrange rows
    """
    di, dj = X.shape
    
    new_X = X.copy()
    new_rows = rows.copy() 
    new_cols = cols.copy() 
    
    # free to move rows
    row_dict = {}
    free_row_idx = np.arange(di)
    linked_rowcol_idx = new_X.argmax(axis=1) # the column with max value for each row
    
    for row, key in zip(free_row_idx, linked_rowcol_idx): 
        if key in row_dict.keys():
            row_dict[key] = row_dict[key] + [row]
        else:
            row_dict[key] = [row]
            
    new_row_order = np.hstack([row_dict[key] for key in sorted(row_dict.keys())])
    # update new_X new_cols
    new_X = new_X[new_row_order, :].copy()
    new_rows = new_rows[new_row_order]
    
    return new_X, new_rows, new_cols 

def partition_network(binary_clst, row_index=np.array([]), col_index=np.array([]), 
                       row_name='row', col_name='col'):
    """Partition a network (binary matrix) into sub-networks 
    """
    groups = []
    row_node, col_node = sparse.lil_matrix(binary_clst).nonzero()
    if row_index.size:
        row_node = [row_index[node] for node in row_node]
    if col_index.size:
        col_node = [col_index[node] for node in col_node]
    nrow, ncol = binary_clst.shape
    if (len(set(row_node)) < nrow) or (len(set(col_node)) < ncol):
        print("Warning: some nodes don't connect to any others!")
    
    for i, j in zip(row_node, col_node):
        if not groups:
            groups.append({row_name: {i}, 
                           col_name: {j},
                          })
        else:
            groupi = -1
            groupj = -1
            for idx, group in enumerate(groups):
                if i in group[row_name]:
                    groupi = idx 
                if j in group[col_name]:
                    groupj = idx 

            # not matched 
            if (groupi==-1 and groupj==-1):
                groups.append({row_name: {i}, 
                               col_name: {j},
                            })
            else:
                if groupi != -1 and groupj != -1 and groupi != groupj:
                    # link 2 groups
                    groups[groupi][row_name] = groups[groupi][row_name].union(groups[groupj][row_name])
                    groups[groupi][col_name] = groups[groupi][col_name].union(groups[groupj][col_name])
                    del groups[groupj]
                    
                else:
                    if groupi != -1:
                        groups[groupi][col_name].add(j)
                    if groupj != -1:
                        groups[groupj][row_name].add(i)
    return pd.DataFrame(groups)

def get_grad_colors(n, cmap='copper'):
    """Generate n colors from a given colormap (a matplotlib.cm)
    """
    from matplotlib import cm
    cmap = cm.get_cmap(cmap)
    return [cmap(int(i)) for i in np.linspace(0, 255, n)] 

def logcpm(counts):
    """
    Args:
        - gene-cell matrix
    """
    cov = counts.sum(axis=0)
    logcpm = np.log10(counts.divide(cov, axis=1)*1000000 + 1)
    return logcpm

def logtpm(counts, gene_lengths):
    """
    Args:
        - gene-cell matrix
        - gene_lengths: a series indexed by gene_id
    """
    tpm = counts.divide(gene_lengths.loc[counts.index], axis=0)
    cov = tpm.sum(axis=0)
    logtpm = np.log10((tpm.divide(cov, axis=1))*1000000 + 1)
    return logtpm

def sparse_logcpm(gc_matrix, mode='logcpm', lib_size=[]):
    """
    """
    lib_size = np.array(lib_size)
    if np.size(lib_size) == 0:
        lib_size = gc_matrix.data.sum(axis=0)

    lib_size_inv = sparse.diags(np.ravel(1.0/(1e-7+lib_size)))
    cpm = (gc_matrix.data).dot(lib_size_inv*1e6).tocoo()

    if mode == 'logcpm':
        cpm.data = np.log10(cpm.data + 1)
    elif mode == 'cpm':
        pass

    gc_cpm = GC_matrix(
        gc_matrix.gene, 
        gc_matrix.cell, 
        cpm,
    )
    
    return gc_cpm

def sparse_logtpm(gc_matrix, gene_lengths):
    """
    gene_lengths: array like 
    
    """
    gene_lengths = np.array(gene_lengths)
    gene_length_inv = sparse.diags(np.ravel(1.0/gene_lengths))
    tmp = (gene_length_inv).dot(gc_matrix.data).tocoo()
    lib_size_inv = sparse.diags(np.ravel(1.0/tmp.sum(axis=0)))
    
    logtpm = tmp.dot(lib_size_inv*1e6).tocoo()
    logtpm.data = np.log10(logtpm.data + 1)

    gc_logtpm = GC_matrix(
        gc_matrix.gene, 
        gc_matrix.cell, 
        logtpm,
    )
    
    return gc_logtpm

def clean_gene_id(gene_id):
    """Remove extra
    """
    return gene_id.split('.')[0]

def gene_id_to_name(gene_id, df_genes):
    """df_genes: gene_id as index, gene_name is one column
    """
    try:
        return df_genes.loc[gene_id, 'gene_name']
    except:
        None

def gene_name_to_id(gene_name, df_genes_v2):
    """df_genes: gene_name as index, gene_id is one column
    """
    try:
        return df_genes_v2.loc[gene_name, 'gene_id'] 
    except:
        return None

def isdataset(dataset):
    """check if a dataset exists
    """
    return os.path.isdir(os.path.join(PATH_DATASETS, dataset))

def isrs2(dataset):
    """check if a dataset is a rs2 dataset
    """
    if dataset.split('_')[1] == 'RS2':
        return True
    else: 
        return False
        
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def create_logger(name='log'):
    """
    args: logger name

    return: a logger object
    """
    logging.basicConfig(
        format='%(asctime)s %(message)s', 
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO)
    return logging.getLogger(name)

def get_sex_from_dataset(dataset):
    """Infer sex info from the name of a dataset
    CEMBA_3C_171206
    CEMA_RS2_Bm3C
    """
    if dataset.split('_')[1] != 'RS2':
        sex = 'M'
    else:
        sex = dataset.split('_')[2][1].upper() 
        if sex not in ['M', 'F']:
            raise ValueError("Sex cannot be infered from dataset name")
    return sex

def slicecode_to_region(slicecode, 
    slicecode_col='code',
    brain_region_col='ABA_acronym',
    reference_table=os.path.join(PATH_REFERENCES, 'Brain_regions', 'RS1_disection_regions.tsv')):
    """Given a slice code, return a brain region (ABA acronym)
    3C -> MOp
    """
    assert len(slicecode) < 4
    slicecode = slicecode.upper()
    df = pd.read_table(reference_table, index_col=slicecode_col)
    try:
        brain_region = df.loc[slicecode, brain_region_col]
    except:
        raise ValueError("Brain region not found!")
    return brain_region
    
def injcode_to_region(injcode, 
    injcode_col='Code',
    brain_region_col='Region name',
    reference_table=os.path.join(PATH_REFERENCES, 'Brain_regions', 'RS2_injection_regions.tsv')):
    """Given a injection code, return a brain region 
    3C -> MOp
    """
    assert len(injcode) == 1 
    injcode = injcode.upper()
    df = pd.read_table(reference_table, index_col=injcode_col)
    try:
        brain_region = df.loc[injcode, brain_region_col]
    except:
        raise ValueError("Brain region not found!")
    return brain_region



def get_mCH_contexts():
    contexts = []
    for base1 in ['A','C','T']:
        for base2 in ['A','C','G','T']:
            contexts.append('C' + base1 + base2)
    return contexts+['CAN', 'CTN', 'CCN']

def get_expanded_context(context):

    if context == "CH":
        # 15 contexts 
        contexts = get_mCH_contexts()
    elif context == "CG":
        contexts = ["CGA","CGC","CGG","CGT","CGN"]
    elif context == "CA":
        contexts = ["CAA","CAC","CAG","CAT","CAN"]
    elif context == "CT":
        contexts = ["CTA","CTC","CTG","CTT","CTN"]
    elif context == "CAG":
        contexts = ["CAG"]
    elif context == "CAC":
        contexts = ["CAC"]
    else:
        raise ValueError('Invalid context.')
    return contexts

def get_chromosomes(species, include_x=True, include_chr=False):
    """
    """
    if species == 'mouse':
        return get_mouse_chromosomes(include_x=include_x, include_chr=include_chr)
    elif species == 'human':
        return get_human_chromosomes(include_x=include_x, include_chr=include_chr)
    else:
        raise ValueError("No such species: {}".format(species))


def get_mouse_chromosomes(include_x=True, include_chr=False):
    chromosomes = [str(x) for x in range(1,20)]
    if include_x:
        chromosomes.append('X')
    if not include_chr:
        return chromosomes
    else:
        return ['chr'+chrom for chrom in chromosomes]

def get_human_chromosomes(include_x=True, include_chr=False):
    chromosomes = [str(x) for x in range(1,23)]
    if include_x:
        chromosomes.append('X')
    if not include_chr:
        return chromosomes
    else:
        return ['chr'+chrom for chrom in chromosomes]

# mm10 
def get_chrom_lengths_mouse(
    genome_size_fname=GENOME_SIZE_FILE_MOUSE):  
    """
    """
    srs_gsize = pd.read_table(genome_size_fname, header=None, index_col=0, squeeze=True)
    srs_gsize = srs_gsize.loc[get_mouse_chromosomes(include_chr=True)]
    # remove leading 'chr'
    srs_gsize.index = [idx[len('chr'):] for idx in srs_gsize.index]
    return srs_gsize

# hg19
def get_chrom_lengths_human(
    genome_size_fname=GENOME_SIZE_FILE_HUMAN):  
    """
    """
    srs_gsize = pd.read_table(genome_size_fname, header=None, index_col=0, squeeze=True)
    srs_gsize = srs_gsize.loc[get_human_chromosomes(include_chr=True)]
    # remove leading 'chr'
    srs_gsize.index = [idx[len('chr'):] for idx in srs_gsize.index]
    return srs_gsize

def tabix_summary(records, context="CH", cap=0):

    mc = 0
    c = 0

    contexts = get_expanded_context(context)

    if cap > 0:
        for record in records:
            if record[3] in contexts:
                if int(record[5]) <= cap:
                    mc += int(record[4])
                    c += int(record[5])
    else:
        for record in records:
            if record[3] in contexts:
                mc += int(record[4])
                c += int(record[5])

    return mc, c


def read_allc_CEMBA(fname, pindex=True, compression='gzip', remove_chr=True, **kwargs):
    """
    """
    if pindex:
        df = pd.read_table(fname, 
            compression=compression,
            header=None, 
            index_col=['chr', 'pos'],
            dtype={'chr': str, 'pos': np.int, 'mc': np.int, 'c': np.int, 'methylated': np.int},
            names=['chr','pos','strand','context','mc','c','methylated'], **kwargs)
    else:
        df = pd.read_table(fname, 
            compression=compression,
            header=None, 
            # index_col=['chr', 'pos'],
            dtype={'chr': str, 'pos': np.int, 'mc': np.int, 'c': np.int, 'methylated': np.int},
            names=['chr','pos','strand','context','mc','c','methylated'], **kwargs)
        
    # remove chr
    if remove_chr:
        if df.iloc[0,0].startswith('chr'):
            df['chr'] = df['chr'].apply(lambda x: x[3:]) 
    return df

def read_genebody(fname, index=True, compression='infer', contexts=CONTEXTS, **kwargs):
    """
    """
    dtype = {'gene_id': object}
    for context in contexts:
        dtype[context] = np.int
        dtype['m'+context] = np.int

    if index:
        df = pd.read_table(fname, 
            compression=compression,
            index_col=['gene_id'],
            dtype=dtype,
            **kwargs
            )
    else:
        df = pd.read_table(fname, 
            compression=compression,
            # index_col=['gene_id'],
            dtype=dtype,
            **kwargs
            )
    return df

def read_binc(fname, index=True, compression='infer', contexts=CONTEXTS, **kwargs):
    """
    """
    dtype = {'chr': object, 'bin': np.int}
    for context in contexts:
        dtype[context] = np.int
        dtype['m'+context] = np.int

    if index:
        df = pd.read_table(fname, 
            compression=compression,
            index_col=['chr', 'bin'],
            dtype=dtype,
            **kwargs
            )
    else:
        df = pd.read_table(fname, 
            compression=compression,
            # index_col=['gene_id'],
            dtype=dtype,
            **kwargs
            )
    return df

def compute_global_mC(dataset, contexts=CONTEXTS, species=SPECIES):
    """return global methylation level as a dataframe indexed by sample
    """

    ti = time.time()
    logging.info('Compute global methylation levels...({}, {})'.format(dataset, contexts))
    dataset_path = os.path.join(PATH_DATASETS, dataset)
    binc_paths = sorted(glob.glob(os.path.join(dataset_path, 'binc/binc_*.bgz')))
    # meta_path = os.path.join(dataset_path, 'mapping_summary_{}.tsv'.format(dataset))

    res_all = []
    for i, binc_file in enumerate(binc_paths):
        if i%100==0:
            logging.info("Progress: {}/{}".format(i+1, len(binc_paths)))

        df = read_binc(binc_file, compression='gzip')
        # filter chromosomes
        df = df[df.index.get_level_values(level=0).isin(get_chromosomes(species))]

        res = {}
        res['sample'] = os.path.basename(binc_file)[len('binc_'):-len('_10000.tsv.bgz')]
        sums = df.sum()
        res['global_mCA'] = sums['mCA']/sums['CA']
        res['global_mCH'] = sums['mCH']/sums['CH']
        res['global_mCG'] = sums['mCG']/sums['CG']
        res_all.append(res) 
        
    df_res = pd.DataFrame(res_all)
    df_res = df_res.set_index('sample')
    tf = time.time()
    logging.info('Done computing global methylation levels, total time spent: {} seconds'.format(tf-ti))
    return df_res


def set_value_by_percentile(this, lo, hi):
    """set `this` below or above percentiles to given values
    this (float)
    lo(float)
    hi(float)
    """
    if this < lo:
        return lo
    elif this > hi:
        return hi
    else:
        return this

def mcc_percentile_norm(mcc, low_p=5, hi_p=95):
    """
    set values above and below specific percentiles to be at the value of percentiles 

    args: mcc, low_p, hi_p  

    return: normalized mcc levels
    """
#   mcc_norm = [np.isnan(mcc) for mcc_i in list(mcc)]
    mcc_norm = np.copy(mcc)
    mcc_norm = mcc_norm[~np.isnan(mcc_norm)]

    lo = np.percentile(mcc_norm, low_p)
    hi = np.percentile(mcc_norm, hi_p)

    mcc_norm = [set_value_by_percentile(mcc_i, lo, hi) for mcc_i in list(mcc)]
    mcc_norm = np.array(mcc_norm)

    return mcc_norm

def plot_tsne_values(df, tx='tsne_x', ty='tsne_y', tc='mCH',
                    low_p=5, hi_p=95,
                    s=2,
                    cbar_label=None,
                    output=None, show=True, close=False, 
                    t_xlim='auto', t_ylim='auto', title=None, figsize=(8,6), **kwargs):
    """
    tSNE plot

    xlim, ylim is set to facilitate displaying glial clusters only

    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.scatter(df[tx], df[ty], s=s, 
        c=mcc_percentile_norm(df[tc].values, low_p=low_p, hi_p=hi_p), **kwargs)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(tc)
    ax.set_xlabel(tx)
    ax.set_ylabel(ty)
    # ax.set_aspect('auto')


    clb = plt.colorbar(im, ax=ax)
    if cbar_label:
        clb.set_label(cbar_label, rotation=270, labelpad=10)

    if t_xlim == 'auto':
        t_xlim = [np.nanpercentile(df[tx].values, 0.1), np.nanpercentile(df[tx].values, 99.9)]
        t_xlim[0] = t_xlim[0] - 0.1*(t_xlim[1] - t_xlim[0])
        t_xlim[1] = t_xlim[1] + 0.1*(t_xlim[1] - t_xlim[0])
        ax.set_xlim(t_xlim)
    elif t_xlim:
        ax.set_xlim(t_xlim)
    else:
        pass  

    if t_ylim == 'auto':
        t_ylim = [np.nanpercentile(df[ty].values, 0.1), np.nanpercentile(df[ty].values, 99.9)]
        t_ylim[0] = t_ylim[0] - 0.1*(t_ylim[1] - t_ylim[0])
        t_ylim[1] = t_ylim[1] + 0.1*(t_ylim[1] - t_ylim[0])
        ax.set_ylim(t_ylim)
    elif t_ylim:
        ax.set_ylim(t_ylim)
    else:
        pass

    fig.tight_layout()
    if output:
        fig.savefig(output)
        print('Saved to ' + output) 
    if show:
        plt.show()
    if close:
        plt.close(fig)

def tsne_and_boxplot(df, tx='tsne_x', ty='tsne_y', tc='mCH', bx='cluster_ID', by='mCH',
                    output=None, show=True, close=False, title=None, figsize=(6,8),
                    t_xlim='auto', t_ylim='auto', b_xlim=None, b_ylim='auto', 
                    low_p=5, hi_p=95):
    """
    boxplot and tSNE plot

    xlim, ylim is set to facilitate displaying glial clusters only

    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axs = plt.subplots(2,1,figsize=figsize)

    ax = axs[0]
    im = ax.scatter(df[tx], df[ty], s=2, 
        c=mcc_percentile_norm(df[tc].values, low_p=low_p, hi_p=hi_p))
    # ax.set_xlim([-40, 40])
    # ax.set_ylim([40, 100])
    if title:
        ax.set_title(title)
    else:
        ax.set_title(tc)
    ax.set_xlabel('tsne_x')
    ax.set_ylabel('tsne_y')
    # ax.set_aspect('auto')
    clb = plt.colorbar(im, ax=ax)
    clb.set_label(tc, rotation=270, labelpad=10)

    if t_xlim == 'auto':
        t_xlim = [np.nanpercentile(df[tx].values, 0.1), np.nanpercentile(df[tx].values, 99.9)]
        t_xlim[0] = t_xlim[0] - 0.1*(t_xlim[1] - t_xlim[0])
        t_xlim[1] = t_xlim[1] + 0.1*(t_xlim[1] - t_xlim[0])
        ax.set_xlim(t_xlim)
    elif t_xlim:
        ax.set_xlim(t_xlim)
    else:
        pass  
    if t_ylim == 'auto':
        t_ylim = [np.nanpercentile(df[ty].values, 0.1), np.nanpercentile(df[ty].values, 99.9)]
        t_ylim[0] = t_ylim[0] - 0.1*(t_ylim[1] - t_ylim[0])
        t_ylim[1] = t_ylim[1] + 0.1*(t_ylim[1] - t_ylim[0])
        ax.set_ylim(t_ylim)
    elif t_ylim:
        ax.set_ylim(t_ylim)
    else:
        pass
 

    ax = axs[1]
    sns.boxplot(x=bx, y=by, data=df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    if b_ylim == 'auto':
        b_ylim = [np.nanpercentile(df[by].values, 1), np.nanpercentile(df[by].values, 99)]
        b_ylim[0] = b_ylim[0] - 0.1*(b_ylim[1] - b_ylim[0])
        b_ylim[1] = b_ylim[1] + 0.1*(b_ylim[1] - b_ylim[0])
        ax.set_ylim(b_ylim)
    elif t_ylim:
        ax.set_ylim(b_ylim)
    else:
        pass

    fig.tight_layout()
    if output:
        # output_dir = './preprocessed/marker_genes_%s' % method
        # if not os.path.exists(output_dir):
        #   os.makedirs(output_dir)

        # output_fname = os.path.join(output_dir, '%s_%s.pdf' % (cluster_ID, gene_name))
        fig.savefig(output)
        print('Saved to ' + output) 
    if show:
        plt.show()
    if close:
        plt.close(fig)


def get_kwcolors(labels, colors):
    """Generate a dictinary of {label: color} using unique labels and a list of availabel colors
    """
    nc = len(colors)
    nl = len(labels)
    n_repeats = int((nl + nc - 1)/nc)
    colors = list(colors)*n_repeats
    
    kw_colors = {l:c for (l,c) in zip(labels, colors)}
    return kw_colors

def rgb2hex(r,g,b):
    """From rgb (255, 255, 255) to hex
    """
    hex = "#{:02x}{:02x}{:02x}".format(int(r),int(g),int(b))
    return hex

def gen_colors(n, l=0.6, s=0.6, colors=None):
    """Generate compatible and distinct hex colors
    """
    if not colors:
        import colorsys
        hs = np.linspace(0, 1, n, endpoint=False)
        rgbs = [rgb2hex(*(256*np.array(colorsys.hls_to_rgb(h, l, s))))
                 for h in hs]
        return rgbs
    else:
        clrs = [colors[i%len(colors)] for i in range(n)] 
        return clrs 

def myScatter(ax, df, x, y, l, 
              s=20,
              sample_frac=None,
              sample_n=None,
              legend_size=None,
              legend_kws=None,
              grey_label='unlabeled',
              shuffle=True,
              random_state=None,
              legend_mode=0,
              kw_colors=False,
              colors=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C8', 'C9'], **kwargs):
    """
    take an axis object and make a scatter plot

    - kw_colors is a dictinary {label: color}
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    df = df.copy()
    # shuffle (and copy) data
    if sample_n:
        df = (df.groupby(l).apply(lambda x: x.sample(min(len(x), sample_n), random_state=random_state))
                            .reset_index(level=0, drop=True)
            )
    if sample_frac:
        df = (df.groupby(l).apply(lambda x: x.sample(frac=sample_frac, random_state=random_state))
                            .reset_index(level=0, drop=True)
            )
    if shuffle:
        df = df.sample(frac=1, random_state=random_state)

    if not kw_colors:
        # add a color column
        inds, catgs = pd.factorize(df[l])
        df['c'] = [colors[i%len(colors)] if catgs[i]!=grey_label else 'grey' 
                    for i in inds]
    else:
        df['c'] = [kw_colors[i] if i!=grey_label else 'grey' for i in df[l]]
    
    # take care of legend
    if legend_mode != -1:
        for ind, row in df.groupby(l).first().iterrows():
            ax.scatter(row[x], row[y], c=row['c'], label=ind, s=s, **kwargs)
        
    if legend_mode == -1:
        pass
    elif legend_mode == 0:
        lgnd = ax.legend()
    elif legend_mode == 1:
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        lgnd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
              ncol=6, fancybox=False, shadow=False) 
    elif legend_mode == 2:
        # Shrink current axis's width by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0 + box.width*0.1, box.y0,
                                 box.width*0.8, box.height])

    if legend_kws:
        lgnd = ax.legend(**legend_kws)

    if legend_mode != -1 and legend_size:
        for handle in lgnd.legendHandles:
            handle._sizes = [legend_size] 

    # backgroud (grey)
    df_grey = df.loc[df['c']=='grey']
    if not df_grey.empty:
        ax.scatter(df_grey[x], 
                   df_grey[y],
                   c=df_grey['c'], s=s, **kwargs)
    # actual plot
    df_tmp = df.loc[df['c']!='grey']
    ax.scatter(df_tmp[x], 
               df_tmp[y],
               c=df_tmp['c'], s=s, **kwargs)
    
    return

def plot_tsne_labels_ax(df, ax, tx='tsne_x', ty='tsne_y', tc='cluster_ID', 
                    sample_frac=None,
                    sample_n=None,
                    legend_size=None,
                    legend_kws=None,
                    grey_label='unlabeled',
                    legend_mode=0,
                    s=1,
                    shuffle=True,
                    random_state=None,
                    t_xlim='auto', t_ylim='auto', title=None, 
                    legend_loc='lower right',
                    colors=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C8', 'C9'], **kwargs):
    """
    tSNE plot

    xlim, ylim is set to facilitate displaying glial clusters only

    # avoid gray-like 'C7' in colors
    # color orders are arranged for exci-inhi-glia plot 11/1/2017
    """
    import matplotlib.pyplot as plt

    myScatter(ax, df, tx, ty, tc,
             s=s,
             sample_frac=sample_frac,
             sample_n=sample_n,
             legend_size=legend_size,
             legend_kws=legend_kws,
             shuffle=shuffle,
             grey_label=grey_label,
             random_state=random_state, 
             legend_mode=legend_mode, 
             colors=colors, **kwargs)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(tc)
    ax.set_xlabel(tx)
    ax.set_ylabel(ty)
    # ax.set_aspect('auto')

    if t_xlim == 'auto':
        t_xlim = [np.nanpercentile(df[tx].values, 0.1), np.nanpercentile(df[tx].values, 99.9)]
        t_xlim[0] = t_xlim[0] - 0.1*(t_xlim[1] - t_xlim[0])
        t_xlim[1] = t_xlim[1] + 0.1*(t_xlim[1] - t_xlim[0])
        ax.set_xlim(t_xlim)
    elif t_xlim:
        ax.set_xlim(t_xlim)
    else:
        pass  

    if t_ylim == 'auto':
        t_ylim = [np.nanpercentile(df[ty].values, 0.1), np.nanpercentile(df[ty].values, 99.9)]
        t_ylim[0] = t_ylim[0] - 0.1*(t_ylim[1] - t_ylim[0])
        t_ylim[1] = t_ylim[1] + 0.1*(t_ylim[1] - t_ylim[0])
        ax.set_ylim(t_ylim)
    elif t_ylim:
        ax.set_ylim(t_ylim)
    else:
        pass

    return


def plot_tsne_labels(df, tx='tsne_x', ty='tsne_y', tc='cluster_ID', 
                    grey_label='unlabeled',
                    sample_frac=None,
                    sample_n=None,
                    legend_size=None,
                    legend_mode=0,
                    legend_kws=None,
                    s=1,
                    random_state=None,
                    output=None, show=True, close=False, 
                    t_xlim='auto', t_ylim='auto', title=None, figsize=(8,6),
                    legend_loc='lower right',
                    colors=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C8', 'C9'], **kwargs):
    """
    tSNE plot

    xlim, ylim is set to facilitate displaying glial clusters only

    # avoid gray-like 'C7' in colors
    # color orders are arranged for exci-inhi-glia plot 11/1/2017
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=figsize)

    myScatter(ax, df, tx, ty, tc,
             s=s,
             sample_frac=sample_frac,
             sample_n=sample_n,
             legend_size=legend_size,
             legend_kws=legend_kws,
             grey_label=grey_label,
             random_state=random_state, 
             legend_mode=legend_mode, 
             colors=colors, **kwargs)

    if title:
        ax.set_title(title)
    else:
        ax.set_title(tc)
    ax.set_xlabel(tx)
    ax.set_ylabel(ty)
    # ax.set_aspect('auto')

    if t_xlim == 'auto':
        t_xlim = [np.nanpercentile(df[tx].values, 0.1), np.nanpercentile(df[tx].values, 99.9)]
        t_xlim[0] = t_xlim[0] - 0.1*(t_xlim[1] - t_xlim[0])
        t_xlim[1] = t_xlim[1] + 0.1*(t_xlim[1] - t_xlim[0])
        ax.set_xlim(t_xlim)
    elif t_xlim:
        ax.set_xlim(t_xlim)
    else:
        pass  

    if t_ylim == 'auto':
        t_ylim = [np.nanpercentile(df[ty].values, 0.1), np.nanpercentile(df[ty].values, 99.9)]
        t_ylim[0] = t_ylim[0] - 0.1*(t_ylim[1] - t_ylim[0])
        t_ylim[1] = t_ylim[1] + 0.1*(t_ylim[1] - t_ylim[0])
        ax.set_ylim(t_ylim)
    elif t_ylim:
        ax.set_ylim(t_ylim)
    else:
        pass

    if output:
        fig.savefig(output)
        print('Saved to ' + output) 
    if show:
        plt.show()
    if close:
        plt.close(fig)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def compress(file, suffix='bgz'):
    """
    """
    # compress and name them .bgz
    try:
        sp.run("bgzip -f {}".format(file), shell=True)
        if suffix != 'gz':
            sp.run("mv {}.gz {}.{}".format(file, file, suffix), shell=True)
    except:
        sp.call("bgzip -f {}".format(file), shell=True)
        if suffix != 'gz':
            sp.call("mv {}.gz {}.{}".format(file, file, suffix), shell=True)
    return

def get_cluster_mc_c(ens, context, genome_regions='bin', 
                     cluster_col='cluster_mCHmCG_lv_npc50_k30', database=DATABASE):
    """Example arguments:
    - ens: 'Ens1'
    - context: 'CG'
    - genome_regions: 'bin' or 'genebody'
    - cluster_col: 'cluster_mCHmCG_lv_npc50_k30'
    - database: 'CEMBA'
    """
    from CEMBA_update_mysql import connect_sql
    from CEMBA_init_ensemble_v2 import pull_genebody_info
    from CEMBA_init_ensemble_v2 import pull_binc_info
    
    ens_path = os.path.join(PATH_ENSEMBLES, ens)
    engine = connect_sql(database) 
    sql = """SELECT * FROM {}
            JOIN cells
            ON {}.cell_id = cells.cell_id""".format(ens, ens)
    df_cells = pd.read_sql(sql, engine) 
    cells = df_cells.cell_name.values
    
    if genome_regions == 'bin':

        input_f = os.path.join(ens_path, 'binc/binc_m{}_100000_{}.tsv.bgz'.format(context, ens)) 
        if not os.path.isfile(input_f):
            ###!!! This part of code is not tested and subject to errors!
            logging.info("Unable to find bin*cell matrix in {}, pulling info from datasets".format(input_f))

            ens_binc_path = os.path.join(ens_path, 'binc')
            binc_paths = [os.path.join(PATH_DATASETS, dataset, 'binc', 'binc_{}_{}.tsv.bgz'.format(cell, BIN_SIZE)) 
                      for (cell, dataset) in zip(df_cells.cell_name, df_cells.dataset)]

            dfs_gb, contexts = pull_binc_info(ens, ens_binc_path, cells, binc_paths, 
                            contexts=CONTEXTS, to_file=True)
            df_input = dfs_gb[contexts.index(context)]
        else:
            logging.info("Found bin*cell matrix in {}".format(input_f))
            # binc
            df_input = pd.read_table(input_f, 
                index_col=['chr', 'bin'], dtype={'chr': object}, compression='gzip')
        
    elif genome_regions == 'genebody':

        input_f = os.path.join(ens_path, 'gene_level/genebody_m{}_{}.tsv.bgz'.format(context, ens)) 
        if not os.path.isfile(input_f):
            logging.info("Unable to find gene*cell matrix in {}, pulling info from datasets".format(input_f))

            ens_genelevel_path = os.path.join(ens_path, 'gene_level')
            genebody_paths = [os.path.join(PATH_DATASETS, dataset, 'gene_level', 'genebody_{}.tsv.bgz'.format(cell)) 
                      for (cell, dataset) in zip(df_cells.cell_name, df_cells.dataset)]

            dfs_gb, contexts = pull_genebody_info(ens, ens_genelevel_path, cells, genebody_paths, 
                            contexts=CONTEXTS, to_file=True)
            df_input = dfs_gb[contexts.index(context)]
        else:
            logging.info("Found gene*cell matrix in {}".format(input_f))
            df_input = pd.read_table(input_f, 
                index_col=['gene_id'], compression='gzip')

    else: 
        raise ValueError("Invalid input genome_regions, choose from 'bin' or 'genebody'")

    # cluster mc_c
    df_c = df_input.filter(regex='_c$')
    df_mc = df_input.filter(regex='_mc$')

    df_mc_c = pd.DataFrame() 
    for label, df_sub in df_cells.groupby('{}'.format(cluster_col)):
        samples = df_sub['cell_name'].values
        df_mc_c['cluster_{}_mc'.format(label)] = df_mc[samples+'_mc'].sum(axis=1)
        df_mc_c['cluster_{}_c'.format(label)] = df_c[samples+'_c'].sum(axis=1)

    logging.info("Output shape: {}".format(df_mc_c.shape))
    return df_mc_c


def plot_tsne_values_ax(df, ax, tx='tsne_x', ty='tsne_y', tc='mCH',
                    low_p=5, hi_p=95,
                    s=2,
                    cbar=True,
                    cbar_ax=None,
                    cbar_label=None,
                    t_xlim='auto', t_ylim='auto', title=None, **kwargs):
    """
    tSNE plot

    xlim, ylim is set to facilitate displaying glial clusters only

    """
    import matplotlib.pyplot as plt


    im = ax.scatter(df[tx], df[ty], s=s, 
        c=mcc_percentile_norm(df[tc].values, low_p=low_p, hi_p=hi_p), **kwargs)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(tc)
    # ax.set_aspect('auto')
    if cbar:
        if cbar_ax:
            clb = plt.colorbar(im, cax=cbar_ax, shrink=0.4)
        else:
            clb = plt.colorbar(im, cax=ax, shrink=1)
        if cbar_label:
            clb.set_label(cbar_label, rotation=270, labelpad=10)

    if t_xlim == 'auto':
        t_xlim = [np.nanpercentile(df[tx].values, 0.1), np.nanpercentile(df[tx].values, 99.9)]
        t_xlim[0] = t_xlim[0] - 0.1*(t_xlim[1] - t_xlim[0])
        t_xlim[1] = t_xlim[1] + 0.1*(t_xlim[1] - t_xlim[0])
        ax.set_xlim(t_xlim)
    elif t_xlim:
        ax.set_xlim(t_xlim)
    else:
        pass  

    if t_ylim == 'auto':
        t_ylim = [np.nanpercentile(df[ty].values, 0.1), np.nanpercentile(df[ty].values, 99.9)]
        t_ylim[0] = t_ylim[0] - 0.1*(t_ylim[1] - t_ylim[0])
        t_ylim[1] = t_ylim[1] + 0.1*(t_ylim[1] - t_ylim[0])
        ax.set_ylim(t_ylim)
    elif t_ylim:
        ax.set_ylim(t_ylim)
    else:
        pass

    return im  


def get_mcc(df, base_call_cutoff=100, sufficient_coverage_fraction=1, suffix=True, fillna=True):
    """Get mcc matrix from mc_c matrix (filtering out low coverage gene or bins)
    """
    logging.info('Getting mcc matrix from mc and c') 
    logging.info('base_call_cutoff={}, sufficient_coverage_fraction={}'.format(
                base_call_cutoff, sufficient_coverage_fraction))
    
    df_c = df.filter(regex="_c$")
    df_c.columns = [col[:-len('_c')] for col in df_c.columns] 
    df_mc = df.filter(regex="_mc$")
    df_mc.columns = [col[:-len('_mc')] for col in df_mc.columns] 
    # a gene is sufficiently covered in % of cells 
    condition = (df_c > base_call_cutoff).sum(axis=1) >= sufficient_coverage_fraction*(df.shape[1])/2.0 

    logging.info("Matrix size before pruning... "+ str(df.shape))
    logging.info("Matrix size after pruning... "+ str(df.loc[condition].shape))
    
    # get mcc matrix with kept bins and nan values for low coverage sites
    df_c_nan = df_c.copy()
    df_c_nan[df_c < base_call_cutoff] = np.nan
    df_mcc = df_mc.loc[condition]/df_c_nan.loc[condition]
    logging.info(df_mcc.shape)

    # imputation (missing value -> mean value of all cells)
    if fillna:
        logging.info('Imputing data... (No effect if sufficient_coverage_fraction=1)')
        means = df_mcc.mean(axis=1)
        fill_value = pd.DataFrame({col: means for col in df_mcc.columns})
        df_mcc.fillna(fill_value, inplace=True)
    
    # add suffix
    if suffix:
        df_mcc.columns = df_mcc.columns.values + '_mcc'
    
    return df_mcc

def get_mcc_lite(mc_table, c_table, base_call_cutoff=100, sufficient_coverage_fraction=1, fillna=True):
    """Given 2 numpy array, return mcc table
    Gene/region by sample matrix
    """
    df_c = pd.DataFrame(c_table)
    df_mc = pd.DataFrame(mc_table)
    assert df_c.shape == df_mc.shape
    
    # a gene is sufficiently covered in % of cells 
    condition = (df_c > base_call_cutoff).sum(axis=1) >= sufficient_coverage_fraction*(df_c.shape[1])

    logging.info("Matrix size before pruning... "+ str(df_c.shape))
    logging.info("Matrix size after pruning... "+ str(df_c.loc[condition].shape))
    
    # get mcc matrix with kept bins and nan values for low coverage sites
    df_c_nan = df_c.copy()
    df_c_nan[df_c < base_call_cutoff] = np.nan
    df_mcc = df_mc.loc[condition]/df_c_nan.loc[condition]
    logging.info(df_mcc.shape)

    # imputation (missing value -> mean value of all cells)
    if fillna:
        logging.info('Imputing data... (No effect if sufficient_coverage_fraction=1)')
        means = df_mcc.mean(axis=1)
        fill_value = pd.DataFrame({col: means for col in df_mcc.columns})
        df_mcc.fillna(fill_value, inplace=True)
    
    # return matrix and index (regions)
    return df_mcc.values, df_mcc.index.values

def get_mcc_lite_v2(df_c, df_mc, base_call_cutoff):
    """
    """
    # get mcc matrix with kept bins and nan values for low coverage sites
    df_c_nan = df_c.copy()
    df_c_nan[df_c < base_call_cutoff] = np.nan
    df_mcc = df_mc/df_c_nan
    logging.info(df_mcc.shape)

    # imputation (missing value -> mean value of all cells)
    means = df_mcc.mean(axis=1)
    fill_value = pd.DataFrame({col: means for col in df_mcc.columns})
    df_mcc.fillna(fill_value, inplace=True)
    
    return df_mcc

def get_mcc_lite_v3(df_c, df_mc, base_call_cutoff):
    """
    """
    # get mcc matrix with kept bins and nan values for low coverage sites
    df_c_nan = df_c.copy()
    df_c_nan[df_c < base_call_cutoff] = np.nan
    df_mcc = df_mc/df_c_nan
    return df_mcc


def get_clusters_mc_c_worker(df_cells, df_input, cluster_col):
    """reduce gene*cell or bin*cell matrix to a gene*cluster or bin*cluster matrix
    Arguments:
        - df_cells: a dataframe indexed by 'cell_name', and have '$cluster_col' as column
        - df_input: a dataframe with 'sample_mc', 'sample_c' ... as columns
        sample names are cell names
    """
    # cluster mc_c
    df_c = df_input.filter(regex='_c$')
    df_mc = df_input.filter(regex='_mc$')

    df_mc_c = pd.DataFrame() 
    for label, df_sub in df_cells.groupby(cluster_col):
        samples = df_sub.index.values
        df_mc_c['{}_mc'.format(label)] = df_mc[samples+'_mc'].sum(axis=1)
        df_mc_c['{}_c'.format(label)] = df_c[samples+'_c'].sum(axis=1)

    logging.info("Output shape: {}".format(df_mc_c.shape))
    return df_mc_c


def pull_genebody_mc_c(ens, context, database=DATABASE):
    """Example arguments:
    - ens: 'Ens1'
    - context: 'CG'
    - database: 'CEMBA'
    """
    from CEMBA_update_mysql import connect_sql
    from CEMBA_init_ensemble_v2 import pull_genebody_info
    from CEMBA_init_ensemble_v2 import pull_binc_info
    
    ens_path = os.path.join(PATH_ENSEMBLES, ens)
    engine = connect_sql(database) 
    sql = """SELECT * FROM {}
            JOIN cells
            ON {}.cell_id = cells.cell_id""".format(ens, ens)
    df_cells = pd.read_sql(sql, engine) 
    cells = df_cells.cell_name.values
    
    input_f = os.path.join(ens_path, 'gene_level/genebody_m{}_{}.tsv.bgz'.format(context, ens)) 
    if not os.path.isfile(input_f):
        logging.info("Unable to find gene*cell matrix in {}, pulling info from datasets".format(input_f))

        ens_genelevel_path = os.path.join(ens_path, 'gene_level')
        genebody_paths = [os.path.join(PATH_DATASETS, dataset, 'gene_level', 'genebody_{}.tsv.bgz'.format(cell)) 
                  for (cell, dataset) in zip(df_cells.cell_name, df_cells.dataset)]

        dfs_gb, contexts = pull_genebody_info(ens, ens_genelevel_path, cells, genebody_paths, 
                        contexts=CONTEXTS, to_file=True)
        df_input = dfs_gb[contexts.index(context)]
    else:
        logging.info("Found gene*cell matrix in {}".format(input_f))
        df_input = pd.read_table(input_f, 
            index_col=['gene_id'], compression='gzip')

    logging.info("Output shape: {}".format(df_input.shape))
    return df_input


def rank_array(array):
    """Return ranking of each element of an array
    """
    array = np.array(array)
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks

# added 4/5/2019
def rank_rows(matrix):
    """Return rankings of each rwo in a 2d array
    """
    matrix = np.array(matrix)
    return np.apply_along_axis(rank_array, 1, matrix) # row = 1

def spearman_corrcoef(X, Y):
    """return spearman correlation matrix for each pair of rows of X and Y
    """
    return np.corrcoef(rank_rows(X), rank_rows(Y))

def spearmanr_paired_rows(X, Y):
    from scipy import stats
    
    X = np.array(X)
    Y = np.array(Y)
    corrs = []
    ps = []
    for x, y in zip(X, Y):
        r, p = stats.spearmanr(x, y)
        corrs.append(r)
    return np.array(corrs), np.array(ps)

def get_index_from_array(arr, inqs, na_rep=-1):
    """Get index of array
    """
    arr = np.array(arr)
    arr = pd.Series(arr).reset_index().set_index(0)
    idxs = arr.reindex(inqs)['index'].fillna(na_rep).astype(int).values
    return idxs

def get_genomic_distance(sa, ea, sb, eb):
    """Get genomic distance
    """
    assert sa < ea and sb < eb
    if sa > sb:
        sa, sb = sb, sa
        ea, eb = eb, ea
        
    # sa <= sb
    distance = max(0, sb - ea)
    
    return distance

def get_reverse_comp(string):
    """Get reverse compliment of a string
    """
    comp_dict = {
        'A': 'T',
        'T': 'A',
        'G': 'C',
        'C': 'G',
        'N': 'N',
    }
    for char in set(string):
        if char not in ['A', 'C', 'G', 'T', 'N']:
            raise ValueError('Not allowed char in string')
            
    new_string = ''.join([comp_dict[char] for char in string[::-1]])
    return new_string
    
# added 4/11/2019
def save_gxc_matrix(gxc, f_mat, f_gene, f_cell):
    """
    """
    print("Deprecated: 6/29/2019 Use save_gc_matrix instead")
    sparse.save_npz(f_mat, gxc.data)
    with open(f_gene, 'w') as f:
        f.write('\n'.join(gxc.gene)+'\n')
    with open(f_cell, 'w') as f:
        f.write('\n'.join(gxc.cell)+'\n')
        
def save_gxc_matrix_methylation(gxc, f_mat_c, f_mat_mc, f_gene, f_cell):
    """
    """
    print("Deprecated: 6/29/2019 Use save_gc_matrix_methylation instead")
    sparse.save_npz(f_mat_c, gxc.data['c'])
    sparse.save_npz(f_mat_mc, gxc.data['mc'])
    with open(f_gene, 'w') as f:
        f.write('\n'.join(gxc.gene)+'\n')
    with open(f_cell, 'w') as f:
        f.write('\n'.join(gxc.cell)+'\n') 

def save_gc_matrix(gc_matrix, f_gene, f_cell, f_mat):
    """
    """
    sparse.save_npz(f_mat, gc_matrix.data)
    with open(f_gene, 'w') as f:
        f.write('\n'.join(gc_matrix.gene)+'\n')
    with open(f_cell, 'w') as f:
        f.write('\n'.join(gc_matrix.cell)+'\n')

def save_gc_matrix_methylation(gc_matrix, f_gene, f_cell, f_mat_mc, f_mat_c):
    """
    """
    sparse.save_npz(f_mat_mc, gc_matrix.data['mc'])
    sparse.save_npz(f_mat_c, gc_matrix.data['c'])
    with open(f_gene, 'w') as f:
        f.write('\n'.join(gc_matrix.gene)+'\n')
    with open(f_cell, 'w') as f:
        f.write('\n'.join(gc_matrix.cell)+'\n') 

def import_single_textcol(fname, header=None, col=0):
    return pd.read_csv(fname, header=header, sep='\t')[col].values

def export_single_textcol(fname, array):
    with open(fname, 'w') as f:
        f.write('\n'.join(array)+'\n')

def load_gc_matrix(f_gene, f_cell, f_mat):
    """
    """
    gene = import_single_textcol(f_gene)
    cell = import_single_textcol(f_cell)
    mat = sparse.load_npz(f_mat) 
    assert (len(gene), len(cell)) == mat.shape
    return GC_matrix(gene, cell, mat) 

def load_gc_matrix_methylation(f_gene, f_cell, f_mat_mc, f_mat_c):
    """
    """
    _gene = import_single_textcol(f_gene) 
    _cell = import_single_textcol(f_cell)
    _mat_mc = sparse.load_npz(f_mat_mc) 
    _mat_c = sparse.load_npz(f_mat_c) 
    gxc_raw = GC_matrix(_gene, _cell, 
                              {'c': _mat_c, 'mc': _mat_mc})
    return gxc_raw

# annoj_URL
def gen_annoj_url(assembly, position, bases, prefix=ANNOJ_URL_PREFIX, file=ANNOJ_URL_FILE):
    """Generate URL for AnnoJ browser view
    """

    url = prefix.strip('/') + '/' + file + '?' + '&'.join(['assembly='+str(assembly), 
                                                            'position='+str(position), 
                                                            'bases='+str(bases)])
    return url


def nondup_legends(ax='', **kwargs):
    """Assuming plt (matplotlib.pyplot) is imported
    """
    from collections import OrderedDict
    import matplotlib.pyplot as plt

    if ax == '':
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), **kwargs)
    else:
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), **kwargs)
    return 

def dedup_array_elements(x, empty_string=''):
    """Replacing repeats with empty_string
    """
    newx = np.empty_like(x)
    newx[0] = x[0]
    for i in range(1, len(x)):
        if x[i-1] == x[i]:
            newx[i] = empty_string
        else:
            newx[i] = x[i]
    return newx

def vcorrcoef(X,Y):
    """Compute correlation coef for each rows of X and Y
    """
    assert X.shape == Y.shape
    Xm = np.mean(X,axis=1).reshape(-1,1)
    Ym = np.mean(Y,axis=1).reshape(-1,1)
    Xm = X-Xm
    Ym = Y-Ym
    
    r_num = np.sum(Xm*Ym,axis=1)
    r_den = np.sqrt(np.sum(Xm**2,axis=1)*np.sum(Ym**2, axis=1))
    r = r_num/r_den
    return r

def zscore(x, offset=1e-7, ddof=1):
    return (x - np.mean(x))/(np.std(x, ddof=ddof) + offset)


def clst_umap_pipe_lite(pcs, cells_all, 
                        resolution=1,
                        npc=50,
                        k=30,
                        verbose=False, seed=0, cluster_only=False, 
                       ):
    # clustering
    import CEMBA_clst_utils
    import CEMBA_run_tsne

    df_clst = CEMBA_clst_utils.clustering_routine(
                                    pcs, 
                                    cells_all, k, 
                                    verbose=verbose,
                                    resolution=resolution,
                                    seed=seed,
                                    metric='euclidean', option='plain', n_trees=10, search_k=-1)

    # umap
    if not cluster_only:
        df_tsne = CEMBA_run_tsne.run_umap_lite(
                    pcs, 
                    cells_all, 
                    verbose=verbose,
                    n_neighbors=30, min_dist=0.5, n_dim=2, 
                    random_state=1)

        df_summary = df_clst.join(df_tsne)
        return df_summary
    else:
        return df_clst

def gen_cdf(array, ax, x_range=[], n_points=1000, show=True, flip=False, **kwargs):
    """
    """
    x = np.sort(array)
    y = np.arange(len(array))/len(array)
    if flip:
        # x = x[::-1]
        y = 1 - y

    if not x_range:
        if show:
            ax.plot(x, y, **kwargs)
        return x, y 
    else:
        start, end = x_range
        xbins = np.linspace(start, end, n_points)
        ybins = np.interp(xbins, x, y)
        if show:
            ax.plot(xbins, ybins, **kwargs)
        return xbins, ybins 

def savefig(fig, path):
    """
    """
    fig.savefig(path, bbox_inches='tight', dpi=300)
    return 