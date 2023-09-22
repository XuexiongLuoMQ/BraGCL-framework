# BraGCL-framework
The paper of WSDM 2024 Demo: **An Interpretable Brain Graph Contrastive Learning Framework for Brain Disease Diagnosis**

# Introduction
We propose an interpretable brain graph contrastive learning framework that aims to learn brain graph representations from insufficient label data for disease prediction and pathogenic analysis. Our framework consists of two key designs: We first use the controllable data augmentation strategy to perturb unimportant structures and attribute features according to node and edge importance scores. Then, considering that the difference between healthy control and patient brain graphs is small, we introduce hard negative sample evaluation to weight negative samples of the contrastive loss, which can learn more discriminative brain graph representations. More importantly, our method can observe salient brain regions and connections to explain prediction results. We conducted disease prediction and interpretable analysis experiments on three publicly available neuroimaging datasets to demonstrate the effectiveness of our framework.

# Framework
![BraGCL-Framework](img/framework.jpg)

# Implement
## Brain Graph Construction
Generating a *Functional/Structural* connectivity from a preprocessed *FMRI/DTI* image. Then construct an adjacent matrix via the K-NN algorithm.
![BraGCL-BuildBrainNetwork](img/BrainNetwor.jpg)

## Pipeline
xxx

# Experiment
## Performance
The proposed method(BraGCL) outperforms baseline methods in three datasets.
![BraGCL-Performance](img/performance.jpg)

## Brain Disorder Analysis
![BraGCL-BDA](img/visulaization.jpg)

# Running
## Requirement
The framework needs the following dependencies:
```
torch~=1.10.2
numpy~=1.22.2
scikit-learn~=1.0.2
scipy~=1.7.3
pandas~=1.4.1
tqdm~=4.62.3
torch-geometric~=2.0.3
torch-cluster 1.5.9
faiss-cpu 1.7.2
```
## Run
To run our model on any of the datasets in our paper, simply run:
```
python main.py --dataset =<dataset name> --modality=<fmri/dti>
```
`--dataset` is the name of the dataset
`--modality` is the type of data, selecting from `fmri` and `dti`
Please place the dataset files in the `data/` folder under the root folder
