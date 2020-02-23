import collections

# meta settings
mods = (
    'snmcseq_gene',
    'snatac_gene',
    'smarter_cells',
    'smarter_nuclei',
    '10x_cells_v2', 
    '10x_cells_v3',
    '10x_nuclei_v3',
    '10x_nuclei_v3_macosko',
    'merfish',
    'epi_retro',
    'patchseq',
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
    'category_col', # neuron or not
    'global_mean', # in general or mch
    'global_mean_mcg', # only for mcg
    'total_reads',
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
    'SubCluster',
    'CH_Rate',
    'CG_Rate',
    'FinalReads',
    '#6DC8BF',
    'mouse',
)

settings_atac = Mod_info(
    mods[1],
    'Open chromatin',
    'atac',
    'tpm', 
    +1, # direction 
    'cell',
    'cluster',
    'cluster',
    'cluster',
    '',
    '',
    '',
    '#00ADDC',
    'mouse',
)

settings_sc = Mod_info(
    mods[2],
    'SmartSeq_cells_AIBS',
    'rna',
    'tpm', 
    +1, # direction 
    'cell',
    'cluster_id',
    'cluster_label',
    'cell_class',
    '',
    '',
    '',
    '#FF3333', # historical reason
    'mouse',
)

settings_sn = Mod_info(
    mods[3],
    'SmartSeq_nuclei_AIBS',
    'rna',
    'tpm', 
    +1, # direction 
    'cell',
    'cluster_id',
    'cluster_label',
    'cell_class',
    '',
    '',
    '',
    '#7D9B3D',
    'mouse',
)

settings_10xc = Mod_info(
    mods[4],
    '10X_cells_v2_AIBS',
    'rna',
    'cpm', 
    +1, # direction 
    'cell',
    'cluster_id',
    'cluster_label',
    'class_label',
    '',
    '',
    '',
    '#F68B1F',
    'mouse',
)

settings_10xc_v3 = Mod_info(
    mods[5],
    '10X_cells_v3_AIBS',
    'rna',
    'cpm', 
    +1, # direction 
    'cell',
    'cluster_id',
    'cluster_label',
    'class_label',
    '',
    '',
    '',
    '#F26739',
    'mouse',
)

settings_10xn_v3 = Mod_info(
    mods[6],
    '10X_nuclei_v3_AIBS',
    'rna',
    'cpm', 
    +1, # direction 
    'cell',
    'cluster_id',
    'cluster_label',
    'class_label',
    '',
    '',
    '',
    '#D4E3A5',
    'mouse',
)

settings_10xn_v3_macosko = Mod_info(
    mods[7],
    '10X_nuclei_v3_Broad',
    'rna',
    'cpm', 
    +1, # direction 
    'cell',
    'cluster_id',
    'cluster_label',
    'class_label',
    '',
    '',
    '',
    '#B6D554',
    'mouse',
)

settings_merfish = Mod_info(
    mods[8],
    'MERFISH',
    'merfish',
    'merfish',
    +1, # direction 
    'cellID',
    'final_label',
    'final_label',
    'final_label',
    '',
    '',
    '',
    '#D65689',
    'mouse',
)

settings_epi_retro = Mod_info(
    mods[9],
    'Epi-retro-seq',
    'mc',
    'mc',
    -1, # direction 
    'cellID',
    'Major Type',
    'Major Type',
    'Major Type',
    '',
    '',
    '',
    '#8341d9',
    'mouse',
)

settings_patch = Mod_info(
    mods[10],
    'Patch-seq',
    'rna',
    'rna',
    +1, # direction 
    'Cell',
    'RNA type',
    'RNA type',
    'RNA type',
    '',
    '',
    '',
    '#b67645',
    'mouse',
)

settings = collections.OrderedDict({
    mods[0]: settings_mc,
    mods[1]: settings_atac,
    mods[2]: settings_sc,
    mods[3]: settings_sn,
    mods[4]: settings_10xc,
    mods[5]: settings_10xc_v3,
    mods[6]: settings_10xn_v3,
    mods[7]: settings_10xn_v3_macosko,
    mods[8]: settings_merfish,
    mods[9]: settings_epi_retro,
    mods[10]: settings_patch,
})

