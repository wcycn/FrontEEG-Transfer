# FrontEEGNet external validation on OpenNeuro ds007169

Eighteen participant-held-out folds use the same two-channel FrontEEGNet architecture after polyphase resampling from 200 to 250 Hz. Both models use the same later target test segments. FrontEEGNet does not use the excluded target prefix; PSD-MLP uses its unlabeled features only to estimate min–max normalization and therefore receives more target-domain information.

| Method | Balanced accuracy | Macro-F1 |
|---|---:|---:|
| FrontEEGNet | 30.16% ± 5.60% | 23.60% ± 5.87% |
| PSD-MLP | 29.77% ± 5.26% | — |

The paired FrontEEGNet minus PSD-MLP difference was 0.39 percentage points (bootstrap 95% CI -2.77 to 3.40; wins/ties/losses 11/0/7).

This is an exploratory evaluation of the architecture on a second participant-disjoint benchmark; it does not transfer MATB weights across tasks or establish superiority over PSD.
