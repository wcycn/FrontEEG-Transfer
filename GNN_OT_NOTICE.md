# GNN + Optimal Transport baseline notice

`scripts/run_matb_ot_gnn_paper.py` and `scripts/run_matb_ot_gnn_fp1_fp2.py` are adapted from the
official notebook accompanying
“Optimal Transport and Graph Neural Networks for Cross-Session Mental Workload
Classification”:

- Source: https://github.com/gdemirezen/gnn-ot-crosssession-eeg
- Copyright (c) 2026 gdemirezen
- License: MIT

The implementation retains the published PSD, graph, backward Sinkhorn OT and
GraphConv design while replacing the notebook's 150-trial Optuna sweep with a
small deterministic validation grid suitable for this course project.

The complete upstream license is retained in `GNN_OT_LICENSE`.
