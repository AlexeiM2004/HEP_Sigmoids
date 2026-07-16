# HEP_Sigmoids

Summer Research Project Repository for ML inference in High Energy Physics

Authors;
- Alexei Maiorov
- Charles Bamsey

Supervisors;
- Ethan Simpson
- Yvonne Peters
- Han 

## Overview

This repository contains the complete work investigating machine learning applications in high energy physics, with a focus on top quark mass reconstruction.

## Project Investigations

### Introduction to Machine Learning in HEP

- Established foundational knowledge of neural networks for physics applications (with penguins)
- Implemented both regression and classification tasks using:
  - Deep Neural Networks (DNNs)
  - Transformer architectures
- Developed understanding of ML pipeline for physics data

### Jet Classification

- Investigated classification of physics jets using multiple architectures:
  - Deep Neural Networks (DNNs)
  - Multi-head attention mechanisms
  - Multi-head Latent Attention (MLA)
- Compared performance across different model architectures

### Top Quark Mass (m_ttbar) Regression

Comprehensive investigation into regressing the invariant mass of the top-antitop system.

#### Direct Mass Regression (Initial Approach)

- Used detector-level inputs: `el_*`, `jet_*`, `mu_*`, and `met_*` branches
- Started with small dataset (0.3 GB)
- Implemented DNNs with iterative improvements:
  - Layer optimization
  - Optimizer tuning 
  - Learning rate scheduling
- Later transitioned to transformer (MHA,MLA) architectures
- Applied feature importance analysis to select most relevant inputs
- Scaled up to larger datasets (1.6 GB)
- Explored advanced techniques including flow matching

#### Kinematic Multi-Regression Approach

- Developed a physics-informed approach predicting full 4-vectors
- Regressed px, py, pz, and energy components for top and antitop
- Derived invariant mass from predicted kinematics
- Compared performance against direct regression

