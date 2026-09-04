# FrontEEGNet nested low-shot target adaptation

All conditions reuse the identical strict-LOSO source checkpoints and nested, balanced, non-overlapping target-S1 calibration windows. Target S3 is used only for final scoring. Repeats are averaged within participant before inference.

Source-only: 54.72% ± 5.46% balanced accuracy.

| Method | Trainable parameters | Labels/class | Balanced accuracy | Difference from source-only | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| Head-only | 51 | 1 | 51.06% ± 6.29% | -3.66 pp | [-5.42, -2.02] pp |
| Head-only | 51 | 2 | 51.63% ± 5.43% | -3.09 pp | [-4.79, -1.47] pp |
| Head-only | 51 | 4 | 52.07% ± 5.21% | -2.65 pp | [-4.19, -1.24] pp |
| Head-only | 51 | 8 | 52.37% ± 5.91% | -2.35 pp | [-4.13, -0.65] pp |
| Head-only | 51 | 16 | 53.31% ± 6.28% | -1.41 pp | [-3.27, +0.32] pp |
| Partial last-block | 803 | 1 | 48.94% ± 5.98% | -5.78 pp | [-7.45, -4.15] pp |
| Partial last-block | 803 | 2 | 50.57% ± 5.83% | -4.15 pp | [-5.99, -2.43] pp |
| Partial last-block | 803 | 4 | 50.54% ± 5.38% | -4.18 pp | [-5.98, -2.63] pp |
| Partial last-block | 803 | 8 | 51.60% ± 5.79% | -3.12 pp | [-5.05, -1.48] pp |
| Partial last-block | 803 | 16 | 52.64% ± 6.43% | -2.08 pp | [-4.10, -0.21] pp |
| Full fine-tuning | 1,915 | 1 | 49.15% ± 6.15% | -5.57 pp | [-7.89, -3.48] pp |
| Full fine-tuning | 1,915 | 2 | 50.28% ± 6.61% | -4.44 pp | [-6.68, -2.51] pp |
| Full fine-tuning | 1,915 | 4 | 51.54% ± 6.86% | -3.18 pp | [-5.66, -1.16] pp |
| Full fine-tuning | 1,915 | 8 | 52.32% ± 7.31% | -2.40 pp | [-5.06, -0.16] pp |
| Full fine-tuning | 1,915 | 16 | 52.82% ± 7.28% | -1.90 pp | [-4.32, +0.16] pp |
| Target-only scratch | 1,915 | 1 | 41.54% ± 7.44% | -13.18 pp | [-16.32, -10.22] pp |
| Target-only scratch | 1,915 | 2 | 43.27% ± 7.11% | -11.45 pp | [-14.05, -8.90] pp |
| Target-only scratch | 1,915 | 4 | 44.35% ± 7.94% | -10.37 pp | [-13.07, -7.87] pp |
| Target-only scratch | 1,915 | 8 | 46.12% ± 8.44% | -8.60 pp | [-11.43, -5.99] pp |
| Target-only scratch | 1,915 | 16 | 47.66% ± 9.77% | -7.07 pp | [-10.60, -3.67] pp |

## Interpretation

Source initialization is useful relative to target-only learning: head-only, partial,
and full fine-tuning all outperform scratch at every budget. However, no adaptation
condition exceeds the 54.72% source-only population mean. The closest result is
head-only adaptation with 16 labels per class (53.31%; paired difference -1.41 pp,
95% CI [-3.27, +0.32]). Zero-calibration FrontEEGNet therefore remains the selected
deployment condition; fine-tuning is retained as a controlled comparison.

## Why can fine-tuning underperform source-only inference?

The observed ordering is consistent with several non-exclusive mechanisms:

1. **High-variance adaptation.** The source model is estimated from 25 participants,
   whereas each target update uses only 3 to 48 labelled windows. A few atypical or
   artifact-contaminated windows can therefore exert disproportionate influence.
2. **Cross-session non-stationarity.** Adaptation uses target S1, while final evaluation
   uses target S3. Fine-tuning can absorb contact, state, amplitude, or noise patterns
   that are specific to S1 and do not persist to S3.
3. **Forgetting and update scope.** Head-only performance approaches source-only as the
   label budget grows, whereas broader updates are more damaging at small budgets. This
   pattern is consistent with partial forgetting of population-stable representations.
4. **Batch-normalization and model selection.** Full fine-tuning updates normalization
   statistics from very small batches. Partial adaptation freezes these statistics and
   still degrades, so normalization instability may contribute but is not sufficient to
   explain the result. The leakage-free fixed-epoch protocol also cannot select a
