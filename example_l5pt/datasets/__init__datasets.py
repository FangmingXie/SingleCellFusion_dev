"""Set up dataset-specific settings
"""

import collections

# meta settings
mods = (
    'snmcseq_gene',
    'smarter_cells',
    'smarter_nuclei',
    '10x_cells_v2', 
)

Mod_info = collections.namedtuple('Mod_info', [
    'mod', 
    'name',
    'mod_category',
    'norm_option',
    'mod_direction', # +1 or -1
    'cell_col',
    'cluster_col', 
    'annot_col', 
    'global_mean', # in general or mch
    'global_mean_mcg', # only for mcg
    'color',
    'species',
])

# settngs
settings_mc = Mod_info(
    mods[0],
    'DNA methylation',
    'mc',
    'mc', 
    -1, # negative direction 
    'cell',
    'SubCluster',
    'SubCluster', #'major_clusters' 'sub_cluster' 
    'CH_Rate',
    'CG_Rate',
    '#6DC8BF',
    'mouse',
)


settings_sc = Mod_info(
    mods[1],
    'SmartSeq_cells_AIBS',
    'rna',
    'tpm', 
    +1, # direction 
    'cell',
    'cluster_id',
    'cluster_label',
    '',
    '',
    '#FF3333', 
    'mouse',
)

settings_sn = Mod_info(
    mods[2],
    'SmartSeq_nuclei_AIBS',
    'rna',
    'tpm', 
    +1, # direction 
    'cell',
    'cluster_id',
    'cluster_label',
    '',
    '',
    '#7D9B3D',
    'mouse',
)

settings_10xc = Mod_info(
    mods[3],
    '10X_cells_v2_AIBS',
    'rna',
    'cpm', 
    +1, # direction 
    'cell',
    'cluster_id',
    'cluster_label',
    '',
    '',
    '#F68B1F',
    'mouse',
)


settings = collections.OrderedDict({
    mods[0]: settings_mc,
    mods[1]: settings_sc,
    mods[2]: settings_sn,
    mods[3]: settings_10xc,
})

