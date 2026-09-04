# Synthetic artifact stress test

The trained 26-fold raw-waveform EEGNet models were evaluated after deterministic
signal-level perturbations were added to each held-out target-session-3 window. Blink
perturbations are common frontal Gaussian pulses (Fp2 amplitude 0.85 times Fp1). Motion
perturbations are channel-asymmetric electrode steps with exponential recovery. The clean
condition exactly reproduces the frozen EEGNet result.

| Condition | Balanced accuracy | Difference from clean | Paired bootstrap 95% CI |
|---|---:|---:|---:|
| Clean | 54.72% ± 5.46% | 0.00 pp | [0.00, 0.00] pp |
| Blink, 50 µV | 49.93% ± 5.89% | −4.79 pp | [−6.35, −3.19] pp |
| Blink, 100 µV | 45.45% ± 4.78% | −9.27 pp | [−11.19, −7.29] pp |
| Blink, 200 µV | 40.38% ± 4.34% | −14.34 pp | [−16.88, −11.72] pp |
| Motion, 25 µV | 52.88% ± 5.93% | −1.84 pp | [−2.76, −0.96] pp |
| Motion, 50 µV | 48.72% ± 5.90% | −6.00 pp | [−7.48, −4.44] pp |
| Motion, 100 µV | 41.60% ± 4.91% | −13.12 pp | [−15.20, −10.91] pp |

All intervals resample the 26 held-out participants, not windows. This experiment is a
controlled robustness test, not an estimate of real-world artifact prevalence or wearable
accuracy. The waveforms are simulations rather than recorded, human-annotated blinks or
head movements. It therefore supports the narrower conclusion that the model is sensitive
to plausible frontal transient contamination and that real-device artifact testing remains
necessary.
