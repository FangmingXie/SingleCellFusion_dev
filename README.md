# SingleCellFusion_dev

SingleCellFusion is a computational tool to integrate single-cell genomics data. This github repo is the development/pilot implementation of this tool. Code in this repo is used in [Luo et al 2019 BioRxiv](https://www.biorxiv.org/content/10.1101/2019.12.11.873398v1) and in [manuscript in preparation](); it is expanded into a [formal implementation](https://github.com/mukamel-lab/SingleCellFusion) of this tool.

For more information:
- [Luo, C. et al. Single nucleus multi-omics links human cortical cell regulatory genome diversity to disease risk variants. bioRxiv 2019.12.11.873398 (2019) doi:10.1101/2019.12.11.873398](https://www.biorxiv.org/content/10.1101/2019.12.11.873398v1)
- [Github repo--mukamel-lab/SingleCellFusion](https://github.com/mukamel-lab/SingleCellFusion)

## Installation
Step 1: Clone this repo.
```bash
git clone https://github.com/FangmingXie/SingleCellFusion_dev.git
cd SingleCellFusion_dev
```

Step 2: Set up a conda environment and install dependent packages. (Skip this step if not needed.)
```bash
conda env create -f environment.yml # create an env named scf_dev
source activate scf_dev
```

## Usage
```./scripts``` contains the main code.

```./example_l5pt``` contains an example of integrating the layer 5 projection-track (L5 PT) neurons from 4 different datasets from the mouse primary motor cortex [manuscript in preparation](). The example includes the organized datasets, code, and results, which could be used as a template for other similar tasks.
