# FrontEEGNet architecture ablation on COG-BCI MATB

All variants use the same 26 strict LOSO folds, source sessions, validation sessions, optimizer, stopping rule and seed as the full model.

| Variant | Parameters | Balanced accuracy | Difference from full | 95% CI |
|---|---:|---:|---:|---:|
| Full FrontEEGNet | 1,915 | 54.72% ± 5.46% | 0.00 pp | — |
| short_temporal | 1,163 | 54.16% ± 5.77% | -0.56 pp | [-1.44, +0.28] pp |
| fixed_common_difference | 1,883 | 54.71% ± 5.56% | -0.01 pp | [-0.94, +0.91] pp |
| fp1_only | 1,899 | 50.85% ± 7.02% | -3.87 pp | [-6.54, -1.01] pp |
| fp2_only | 1,899 | 51.51% ± 7.52% | -3.21 pp | [-6.44, -0.06] pp |

The fixed projection tests learned cross-channel weighting against predefined common/difference components. Single-channel variants test whether both frontal sensors contribute beyond model capacity alone.
