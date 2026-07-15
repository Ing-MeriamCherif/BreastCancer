Overview

This repository contains the full implementation of a multi-task mammography classification pipeline trained and evaluated on the CBIS-DDSM dataset. The work systematically compares backbone architectures and loss-design strategies for simultaneous prediction of six clinical mammography attributes, under consumer-grade GPU constraints representative of low-resource clinical settings.

The paper makes three contributions:


Backbone comparison  CLIP ViT-B/32 vs RadImageNet-pretrained ResNet50, demonstrating that general-purpose contrastive pretraining transfers more robustly than medical-domain pretraining under limited compute
Loss-design ablation  a clinically-motivated loss (class-weighted pathology loss + Gaussian ordinal soft labels + label smoothing) vs. a plain cross-entropy baseline, showing the proposed design shifts the clinical operating point rather than raw discriminative power
Calibration analysis  temperature scaling with ECE reporting, showing that post-hoc calibration is necessary for any clinical deployment of models trained with asymmetric class weights


All experiments were conducted on a single NVIDIA RTX 4050 Laptop GPU (6.4 GB VRAM)  no institutional or cloud compute was used
