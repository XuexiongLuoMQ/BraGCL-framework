# BraGCL-framework
The paper of WSDM 2024 Demo: **An Interpretable Brain Graph Contrastive Learning Framework for Brain Disease Diagnosis**

# Abstract
We propose an interpretable brain graph contrastive learning framework that aims to learn brain graph representations from insufficient label data for disease prediction and pathogenic analysis. Our framework consists of two key designs: We first use the controllable data augmentation strategy to perturb unimportant structures and attribute features according to node and edge importance scores. Then, considering that the difference between healthy control and patient brain graphs is small, we introduce hard negative sample evaluation to weight negative samples of the contrastive loss, which can learn more discriminative brain graph representations. More importantly, our method can observe salient brain regions and connections to explain prediction results. We conducted disease prediction and interpretable analysis experiments on three publicly available neuroimaging datasets to demonstrate the effectiveness of our framework.

# Framework
