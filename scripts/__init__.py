"""Config CEMBA scripts
"""
import time
import logging
import glob
import os
import numpy as np
import pandas as pd
import collections


# define constant variables
CONVENTION = 'CEMBA' # 'CEMBA' or 'human'
SPECIES = 'mouse' # 'human' or 'mouse'

BIN_SIZE = 10000
BIN_SIZE_FEATURE = 10*BIN_SIZE

CONTEXTS = ['CH', 'CG', 'CA']
COMBINED_CONTEXTS_LIST = [['CH', 'CG'], ['CA', 'CG']]

PATH_PROJECT = '/cndd/Public_Datasets/CEMBA/snmCSeq' 
PATH_CEMBA = PATH_PROJECT # For back compatibility

PATH_DATASETS = PATH_PROJECT + '/Datasets'
PATH_ENSEMBLES = PATH_PROJECT + '/Ensembles'
PATH_REFERENCES = PATH_PROJECT + '/References'
PATH_GENEBODY_ANNOTATION = PATH_REFERENCES + '/Annotation/gencode.vM16.annotation_genes.tsv'
GENEBODY = PATH_GENEBODY_ANNOTATION # for back-compatibility
GENOME_SIZE_FILE = PATH_REFERENCES + '/Genome/mm10.chrom.sizes'
GENOME_ALLCG_FILE = PATH_REFERENCES + '/Genome/mm10_all_cg.tsv'

GENOME_SIZE_FILE_HUMAN = '/cndd/Public_Datasets/human_snmcseq/References/Genome/hg19.chrom.sizes' 
GENOME_SIZE_FILE_MOUSE = '/cndd/Public_Datasets/CEMBA/snmCSeq/References/Genome/mm10.chrom.sizes' 

# for auto-annotation
REFERENCE_BINS = (os.path.join(PATH_REFERENCES, 
                                      'Mouse_published/binc_mCH_100000_clusterwise_mcc_mouse_published.tsv'))
# REFERENCE_METADATA = (os.path.join(PATH_REFERENCES, 
#                                   'Human_reference/mapping_summary_Ens0.tsv'))

# for init_ensemble_from_ensembles
FRAC_ENSEMBLES_WITH_BIN = 0.9 # include a bin if it is in nmcc files of more than 90% of singleton ensembles

# tSNE
PERPLEXITIES = [20, 30, 40, 50, 100] 
N_PC = 50 
N_DIM = 2

# louvain
K_NN = [5, 10, 15, 20, 30, 50, 100] 

# dmr
NUM_DMS = 3

# data structures
GC_matrix = collections.namedtuple('GC_matrix', ['gene', 'cell', 'data'])


# mysql
USER = 'f7xie'
HOST = 'brainome'
PWD = '3405040212'
DATABASE = 'CEMBA'
DATABASE_ANNOJ = 'CEMBA_annoj'

CELLS_TABLE_COLS = ['cell_id', 
                     'cell_name', 
                     'dataset', 
                     'NeuN',
                     'cell_type',
                     'global_mCH', 
                     'global_mCG',
                     'global_mCA',
                     'global_mCCC', 
                     'estimated_mCH', 
                     'estimated_mCG',
                     'percent_genome_covered', 
                     'total_reads',
                     'mapped_reads', 
                     'mapping_rate', 
                     'nonclonal_reads', 
                     'percent_nonclonal_rate',
                     'filtered_reads',
                     'filtered_rate',
                     'lambda_mC',
                     ]

DATABASE_ATAC = 'CEMBA_ATAC'
CELLS_TABLE_COLS_ATAC = ['cell_id', 
                     'cell_name', 
                     'dataset', 
                     # 'cell_type',
                     # 'global_mCH', 
                     # 'global_mCG',
                     # 'global_mCA',
                     # 'global_mCCC', 
                     # 'estimated_mCH', 
                     # 'estimated_mCG',
                     # 'percent_genome_covered', 
                     # 'total_reads',
                     # 'mapped_reads', 
                     # 'mapping_rate', 
                     # 'nonclonal_reads', 
                     # 'percent_nonclonal_rate',
                     # 'filtered_reads',
                     # 'filtered_rate',
                     # 'lambda_mC',
                     ]

def rename_ms_cols(column_names):
    """Rename headers in mapping summary files (from Chongyuan) to header names used in mySQL cells table
    """
    dict_rename = {'Sample': 'cell_name', 
                   'Total reads': 'total_reads', 
                   'Mapped reads': 'mapped_reads', 
                   'Mapping rate': 'mapping_rate',
                   'Nonclonal reads': 'nonclonal_reads',
                   '% Nonclonal rate': 'percent_nonclonal_rate', 
                   'Filtered reads': 'filtered_reads', 
                   'Filtered rate': 'filtered_rate', 
                   'Lambda mC/C': 'lambda_mC', 
                   'mCCC/CCC': 'global_mCCC', 
                   'mCG/CG': 'global_mCG', 
                   'mCH/CH': 'global_mCH',
                   'Estimated mCG/CG': 'estimated_mCG', 
                   'Estimated mCH/CH': 'estimated_mCH', 
                   '% Genome covered': 'percent_genome_covered'}
    return [dict_rename[col] for col in column_names] 


# ANNOJ view
ANNOJ_URL_PREFIX = 'https://brainome.ucsd.edu/annoj_private/CEMBA'
ANNOJ_URL_FILE = 'cemba.php'