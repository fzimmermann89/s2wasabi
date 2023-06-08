## Semi-Supervised Learning forSpatially Regularized qMRI Reconstruction - Application to Simultaneous 𝑇_1, 𝐵_0, and 𝐵_1  Mapping 
Source, Weights and Wasabi-MR Sequence (Pulseseq) for #1166, ISMRM 2023

Problem: Training a network to predict T1, B0 and B1 from WASABIT! magnitude images with only 10 slices of volunteer data
### Approach: 
- Pretraining with Synthetic Data
- Finetuning with a combination of
- Supervised Training on Synthetic Data
    - Teacher-Student Training with Seperate Augmentations and Random Recombinations of Parameter Maps
    - Self-Supervised (Masked) Training on In-vivo Data
Results: Noise-robust parameter map estimation with reduced cross-talk between parameter maps

### Content:

#### Sequence:

- WASABI.seq: Pulsesq seq file
- WASABI.py: PyPulsesq-Script for generating the sequence
  
#### Trained models
- UNet-finetunded-724.model: Final UNet
- UNet-pretrained-652.model: UNet after pretraining
- MLP-684.model:  MLP-Baseline trained on synthetic data, as proposed by #2714 ISMRM 2022
 
#### Source Code:
- pretrain.py: Simple skript for pretraining
- fineune.py: ..and finetuning ;)


Unfortunalty, we cannot provide the in-vivo dataset used for fine-tuning.

-- This code will be cleaned up and seperated into the WASABI-specific part and a general example for the training regime for a future publication, 'till then, have fun and feel free to ask questions

(c) Felix Zimmermann, felix.zimmermann@ptb.de
All code is open source licensed under BSD-3 license.

