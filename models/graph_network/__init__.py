"""
models.graph_network — Heterogeneous Graph Neural Network (PyTorch Geometric).

Modules (to be added):
    graph_builder  - Construct patient-disease-drug-lab heterogeneous graphs
    hgt_model      - Heterogeneous Graph Transformer (HGT) implementation
    sage_model     - GraphSAGE baseline model
    trainer        - Training loop with early stopping and LR scheduling
    predictor      - Batch prediction on new patient subgraphs
    metrics        - AUROC, AUPRC, calibration metrics for graph tasks
"""
