# Machine-readable result summaries

This directory contains small, versioned summaries of the completed offline experiments.
Raw public EEG, generated feature arrays, logs and model checkpoints are intentionally not
committed. Every value in these summaries is aggregated at the held-out-participant level;
the scripts under `scripts/` regenerate the full per-participant outputs under `outputs/`.

- `matb_fronteegnet_lowshot_adaptation.json`: source-only, head-only, partial, full and
  target-only-scratch comparisons across five nested label budgets and 26 held-out participants.
- `matb_fronteegnet_architecture_ablation.json`: same-fold channel, temporal-kernel and spatial
  projection ablations for the selected FrontEEGNet configuration.
- `ds007169_eegnet_external_validation.json`: 18-participant four-class replication using the
  same model family.
- `final_fronteegnet_source_training.json`: training metadata for the all-source deployment model.
- `matb_offline_extensions.json`: EEGNet, covariance/alignment, frequency-band and channel
  ablations on the 26-participant strict MATB LOSO protocol.
- `ds007169_external_validation.json`: matched four-band PSD-MLP source-only and low-shot baseline on the 18-participant four-class dataset.
- `matb_synthetic_artifact_stress.json`: deterministic blink/motion corruption sensitivity.
- `device_fronteegnet_partial_evaluation.json`: non-identifying FrontEEGNet evaluation of one
  interrupted real-device self-experiment; incomplete class coverage makes the planned
  three-class endpoint invalid.
