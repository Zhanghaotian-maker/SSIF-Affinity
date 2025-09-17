# SSF-Affinity: Multimodal Deep Learning of Sequence-Structure Features for Precise Protein-Protein Binding Affinity Prediction

## Background

Quantitative prediction of binding affinity in protein-protein interactions (PPIs) is critical for deciphering biological mechanisms and advancing therapeutic antibody development. While experimental methods for measuring binding affinity remain limited by high-cost, low-throughput constraints, deep learning offers a promising alternative. Currently, the accuracy of models based on deep learning for predicting protein-protein complex binding affinity is limited. This study proposes SSF-Affinity, a novel framework that integrates multimodal deep learning strategies to achieve high-precision affinity predictions by fusing global sequence features with local atomic-level structural information.The evaluation of the PDBbind benchmark dataset indicates that SSF-Affinity significantly outperforms existing advanced models. Specifically, this framework effectively reduces redundant computations and noise interference by combining regional atomic selection strategies, and overcomes the limitations of unimodal methods that use only sequence or structural information through a synergistic representation mechanism of sequence and structural features, effectively balancing the contributions of interface interactions and long-range interactions. The case studies on antibody-antigen complexes further validates the model's generalization ability. SSF-Affinity provides a new strategy for predicting binding affinity in protein-protein complexes, offering a new paradigm for AI-driven antibody drug discovery.

## Content

This repository contains codes for training, evaluating and visualizing the SSF-Affinity network. 
This repository will be periodically updated.

## Installation
SSF-Affinity is implemented using Python 3.8.19 and various scientific computing packages (numpy, numba, scikit-learn, torch). We recommend creating a dedicated virtual environment and installing the same package versions, see the full list in requirements.txt.   
This can be done in few lines using virtualenv (https://virtualenv.pypa.io/en/latest/index.html) or Anaconda (https://www.anaconda.com):


With Anaconda:

If you have not previously installed anaconda, install miniconda or anaconda by following these <a href= https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/linux.html> instructions </a>. Once a proper conda environment has been installed and activated, setup SSF-Affinity as follows:

```
conda create -n py_ssf-affinity python=3.8.19
conda activate py_ssf-affinity
```

With virtualenv:
```
virtualenv -p python3.8.19 /path/to/virtualenvs/py_ssf-affinity
source activate
source activate.csh # (for csh shells). 
```

Then:
```
pip install -r requirements.txt
```


## Retraining SSI F-Affinity

The pdb files are publicly available: PDBbind v2020. The dataset is divided into four parts: training set, validation set, test set 1, and test set 2.

### Training from scratch on the Protein-Protein complexes binding affinity data set

This can be done via:

The main stages of train.py are:
1. Download the PDBbind v2020: Protein-protein complexes.

Download PDBbind v2020 from http://www.pdbbind.org.cn/download.php: protein-protein complexes to "datasets/PDBs". 
Exclude those with binding interface sizes that are too large (inappropriate for cells), the presence of non-protein molecules in the files (e.g., DNA), files containing fewer than two chains, and chains involved in binding that cannot be labeled from the dataset. 
To clearly label the interaction partners between chains in PDB protein complexes, identify the chain components of direct pairwise interactions involved in a minimal interaction unit.
Download the full-length amino acid sequences of the chains with direct pairwise interactions for each complex's determined minimal interaction unit from the RCSB PDB website to "datasets/sequences".

2. Data preprocessing

```
cd preprocessing
python preprocessing.py
python get_structure_csv.py
```

3. Graph data encoding and sequence encoding

The node feature of the selected atom within the binding region:

cd ../features
```
python get feature.py node --input dir ../datasets/the_end_CSV --output dir ../based_structure_dataset
```

The edge feature of the selected atom within the bound area:

```
python  get feature.py edge --input dir ../datasets/the_end_PDB --output dir ../based_structure_dataset --max neighbors 10
```

Full-length sequence coding of the directly paired-interacting chain of the smallest interaction unit determined:

```
python get feature.py sequence --input dir ../datasets/sequences --output dir ../sequence_dataset_esm
```
4. Get sequence and structural features of the binding interface

```
python  esmif_structure_features.py 
```

```
python  get interface_feature.py 
```

5. Divide the data set

Divide based_structure_dataset in the 'SSIF-Affinity' directory into based_structure_training_dataset、based_structure_val_dataset、based_structure_test1_dataset and based_structure_test2_dataset.
Divide sequence_dataset_esm in the 'SSIF-Affinity' directory into training_dataset_esm、val_dataset_esm、test1_dataset_esm and test2_dataset_esm.
Divide sequence_dataset_esm in the 'SSIF-Affinity/residue_interface/esm2' directory into training_dataset_esm、val_dataset_esm、test1_dataset_esm and test2_dataset_esm.
Divide structure_dataset_esmif in the 'SSIF-Affinity/residue_interface/esmif' directory into training_dataset_esmif、val_dataset_esmif、test1_dataset_esmif and test2_dataset_esmif.

The folders for all of these datasets are in the home directory

6. Training of the network 

cd ../tools
```
python train.py
```

7. Prediction on the test sets.

cd ..
```
python test.py \
  --model_path result/best_model/best_epoch.pth \
  --affinity_csv test_affinity.csv \
  --esm_dir test_dataset_esm \
  --structure_dir based_structure_test_dataset \
  --interface_seq_dir residue_interface/esm2/test_dataset_esm \
  --interface_struct_dir residue_interface/esmif/test_dataset_esmif \
  --output_csv result/predicted_affinities_test.csv

## Constructing your own data set and model

Use your own dataset (including the  structures of protein-protein complexes, sequences and binding affinity labels) to train and test the network according to the steps mentioned above.


## Contact
For any question and comment regarding the code, please reach out to
23121213864@stu.xidian.edu.cn


## References
If you use this program, please cite our paper:
