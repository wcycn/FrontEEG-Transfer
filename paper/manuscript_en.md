<h1 align="center">FrontEEGNet: Evaluating an EEGNet Adaptation for Cross-Subject Mental Workload Decoding with Two-Channel Wearable EEG</h1>

<p align="center"><strong>Chenyu Wang</strong><br>Zhejiang University</p>

## Abstract

Wearable electroencephalography (EEG) could make passive brain–computer interfaces easier to deploy, but practical headsets often expose only a few channels and differ substantially from laboratory acquisition systems. We study whether task-defined mental workload can be decoded across previously unseen participants when observation is restricted to the frontal Fp1 and Fp2 channels available on a motivating wearable device. We do not propose a new neural-network family. Instead, we adapt the established EEGNet architecture to a two-channel raw-waveform input, call this implementation FrontEEGNet, and evaluate it under deployment-oriented controls. On the COG-BCI Multi-Attribute Task Battery (MATB), 26 participant- and session-disjoint leave-one-subject-out folds separate source training, source-session validation, and held-out-subject testing. FrontEEGNet contains 1,915 trainable parameters and reaches 54.72% ± 5.46% balanced accuracy for three-class workload decoding, exceeding a same-fold power-spectral-density (PSD) multilayer perceptron by 5.57 percentage points (paired-bootstrap 95% confidence interval: 1.91–9.34). Target-S1 fine-tuning consistently outperforms target-only training from scratch, but no head-only, partial, or full-parameter configuration improves the overall mean above zero-calibration inference; the strongest adapted result is 53.31% at 16 labels per class. Single-channel ablations reduce performance by 3.87 points for Fp1 and 3.21 points for Fp2, whereas the paired intervals for shorter temporal kernels and a fixed common/difference spatial projection both include zero. On the four-class OpenNeuro ds007169 n-back dataset, the same architecture retrained from scratch reaches 30.16% ± 5.60%, above the 25% chance level but not significantly above PSD. Synthetic frontal transients reveal substantial sensitivity to blink- and motion-like corruption. Finally, a frozen all-source model is executed on data from an interrupted wearable self-experiment; the acquisition and inference path operates end to end, but the incomplete class coverage precludes a valid three-class device accuracy. Under the fixed public-data protocol, the compact model retains some predictive ability for unseen participants; this does not establish efficacy in wearable use, and low-shot updating, artifact sensitivity, and cross-hardware shift remain unresolved.

## 1 Introduction

Passive brain–computer interfaces infer a user's state without requiring an intentional control command [1]. Mental workload is a relevant target because an estimate of task demand may support adaptive interfaces, training systems, and safety monitoring. EEG is attractive for this purpose because it is non-invasive and provides millisecond-scale measurements. Yet most controlled EEG studies depend on dense caps, conductive preparation, fixed laboratory amplifiers, and trained operators. These requirements limit repeated or everyday use.

Wearable EEG attempts to reduce this burden. Compact hardware can shorten preparation and permit less constrained recording [2], but convenience changes the statistical problem. A small montage discards spatial information, frontal electrodes are strongly exposed to ocular and facial activity, and hardware-specific reference, filtering, gain, contact, and timing can shift the input distribution. Consequently, an algorithm that succeeds on a laboratory recording does not automatically become a reliable wearable system after selecting two nominally matching channels.

This work is motivated by a commercial device that provides EEG at Fp1 and Fp2. The manufacturer did not disclose complete hardware specifications, so we do not analyse proprietary electronics or make claims about the device design beyond observable acquisition behaviour. Instead, we use its available montage to formulate a general question: how much cross-subject workload information remains when an established laboratory EEG algorithm is restricted to two frontal channels, and which failure modes emerge when the resulting pipeline is applied to physical hardware?

We focus on task-defined mental workload. The primary public benchmark, COG-BCI, contains repeated recordings of Easy, Medium, and Difficult MATB conditions [3]. These labels describe controlled task demand rather than a clinical state, an intrinsic cognitive capacity, or a direct measurement of attention. EEG power in theta, alpha, and beta bands has often been associated with workload, but effect direction and reliability vary across tasks and participants [4]. Cross-subject decoding is therefore difficult even before channel and device restrictions are introduced.

The methodological question is not answered by random window splits. EEG distributions vary with anatomy, strategy, baseline state, recording session, and electrode contact. Neighbouring windows from the same participant can be highly correlated, especially when they overlap. We therefore use a participant- and session-disjoint protocol: every test participant is absent from source training and model selection, and the evaluated test recording comes from a session not used for source training. We call the setting with no target-participant data *participant-independent generalization* or source-only inference, and reserve *target-participant adaptation* for parameter updates using labels from target session 1. Both probe transfer-related capability, but they are not treated as the same operation.

We select EEGNet [8] as the main architecture after a controlled comparison with linear, multilayer perceptron (MLP), graph convolutional, covariance, and alignment baselines. EEGNet is already an established compact convolutional model. Our contribution is not its invention. We adapt its input, temporal scale, training protocol, and deployment path for Fp1/Fp2 workload decoding and refer to this concrete implementation as **FrontEEGNet**. The study then asks whether the selected model remains effective across participants, which components matter under a two-channel restriction, whether the architecture generalizes to a second workload benchmark, how it responds to frontal corruption, and what happens when the frozen public-data model meets a physical device.

The contributions are:

1. We formulate and implement FrontEEGNet, a 1,915-parameter two-channel instantiation of EEGNet for 4 s Fp1/Fp2 workload windows, while explicitly retaining EEGNet as the architectural source.
2. We conduct a leakage-controlled 26-fold evaluation that holds out both participant identity and the target recording session, and compare raw-waveform learning with same-fold spectral, graph, covariance, and alignment baselines.
3. We compare zero-calibration inference with nested target-S1 head-only, partial, full, and target-only training to determine whether new-subject labels actually improve the same architecture.
4. We test architectural necessity through single-channel, temporal-kernel, and spatial-projection ablations, then extend evaluation to a second dataset, synthetic perturbations, and a frozen-model physical-device diagnostic.

FrontEEGNet has the highest mean among the methods evaluated under the primary two-channel protocol. The external comparison does not show a clear advantage over PSD, and the device session is insufficient for wearable validation.

![Figure 1. Motivation and evaluation scope for two-channel wearable workload decoding.](figures/figure1_motivation.svg)

**Figure 1. Motivation and evaluation scope for two-channel wearable workload decoding.** (a) Public multi-participant EEG provides source data, while only Fp1 and Fp2 are retained before model training. (b) Wearable deployment combines channel scarcity with unseen users, frontal artifacts, and reference or hardware shift. (c) The evidence chain evaluates zero-calibration and low-shot use through strict MATB leave-one-subject-out testing, external architectural replication on ds007169, and a physical-device diagnostic.

## 2 Related Work

### 2.1 EEG mental-workload decoding

Mental workload reflects an interaction between task demand and individual resources. Physiological measures can respond to workload manipulation, but no single signal is universally valid across tasks [12]. A meta-analysis of EEG spectral measures found workload-related effects across theta, alpha, and beta activity, with frontal theta among the more consistent observations [4]. These findings motivate EEG decoding, but group-level spectral associations do not guarantee participant-independent classification. They also do not establish that a classifier has isolated a neural workload mechanism rather than task-correlated eye, muscle, or behavioural activity.

COG-BCI was released for passive-BCI research and contains more than 100 hours of EEG across cognitive paradigms and repeated sessions [3]. Its MATB subset manipulates three levels of task difficulty through tracking, monitoring, resource-management, and communication demands. The repeated-session structure makes it particularly useful for separating participant generalization from repeated-window recognition.

### 2.2 Cross-subject transfer and alignment

Transfer learning in EEG has been studied across subjects, sessions, devices, and tasks [5]. Classical approaches often transform features or covariance matrices to reduce domain mismatch. Euclidean Alignment whitens trials with a domain-level covariance estimate [6], while Riemannian methods classify covariance representations on the manifold of symmetric positive-definite matrices [7]. These methods are principled, but a two-channel montage provides a very small covariance structure and may limit their spatial advantage.

Deep networks instead learn representations directly from time series. Schirrmeister et al. demonstrated end-to-end convolutional EEG decoding [9], and Lawhern et al. introduced EEGNet as a compact architecture using temporal, depthwise spatial, and separable convolutions [8]. EEGNet is attractive for wearable inference because its parameter and compute requirements are low. Nevertheless, its original design does not guarantee invariance to a new participant, session, or amplifier. Cross-subject validity must be established by the data split rather than inferred from architecture alone.

Demirezen et al. combined graph neural networks and Sinkhorn optimal transport for cross-session workload classification on multichannel COG-BCI recordings [11]. That setting uses a much denser montage and transductive access to unlabeled target-session distributions. Our setting removes the target participant from source training and limits the observation to Fp1/Fp2. Their results therefore motivate relevant baselines but are not directly comparable scores.

### 2.3 Low-channel wearable EEG

Mobile and low-cost EEG systems can retain useful information under favourable conditions [2]. However, reducing a laboratory montage to two electrodes is not equivalent to validating a wearable device. Frontal channels lie close to the eyes, and blinks and eye movements produce large electrical fields that can dominate scalp measurements [13]. With only Fp1/Fp2 and no dedicated electrooculography channel, spatial source separation is severely underdetermined. A high workload score may consequently reflect a mixture of cerebral activity, gaze behaviour, facial tension, contact changes, and genuine workload-related dynamics.

Cross-hardware transfer introduces another shift. Two systems labelled Fp1/Fp2 may differ in reference, electrode material, coupling, analogue filtering, quantization, timing, and declared physical units. We therefore distinguish three levels of evidence: controlled cross-subject performance after simulated channel restriction, replication of the architecture on another public dataset, and diagnostic execution on the target hardware. Only the first two permit complete class-balanced evaluation in the present study.

## 3 Method

### 3.1 Problem formulation

Let \(\mathcal{S}\) be the set of source participants and \(t\) a held-out target participant. A 4 s Fp1/Fp2 window is

$$
x \in \mathbb{R}^{2\times1000},
$$

where both channels are sampled at 250 Hz. The label \(y\in\{0,1,2\}\) denotes Easy, Medium, or Difficult workload. Sessions are indexed by \(r\in\{1,2,3\}\). For every leave-one-subject-out fold, model parameters are estimated from labelled source sessions 1 and 2:

$$
\theta_t^{*}=\arg\min_{\theta}
\sum_{s\in\mathcal{S}\setminus\{t\}}
\sum_{(x,y)\in D_{s,1}\cup D_{s,2}}
-\log p_{\theta}(y\mid x).
$$

Source-participant session 3 is used for early stopping. Generalization is evaluated once on \(D_{t,3}\). No sample from participant \(t\), including sessions 1 and 2, is used to fit FrontEEGNet, normalization statistics, hyperparameters, or checkpoints. This is a zero-shot source-only protocol at the target-participant level.

### 3.2 Signal preprocessing

For COG-BCI, recordings are rereferenced to TP10, band-pass filtered from 1 to 40 Hz, resampled from 500 to 250 Hz, and segmented into 4 s windows with 50% overlap. Only Fp1 and Fp2 are retained. Each window and channel is standardized independently:

$$
\widetilde{x}_{c,:}=\frac{x_{c,:}-\mu(x_{c,:})}
{\sigma(x_{c,:})+\epsilon}.
$$

This operation removes per-window offset and scale and, importantly, requires no target-participant calibration segment. It does not remove ocular or motion artifacts.

For ds007169, released two-channel windows have 800 samples at 200 Hz. We apply polyphase resampling to 250 Hz, yielding the same \(2\times1000\) model input. The architecture is unchanged except for a four-class output layer.

### 3.3 FrontEEGNet

FrontEEGNet is an application-specific EEGNet configuration rather than a new model family. Its computation is summarized in Figure 2 and Table 1. An input of shape \(B\times2\times1000\) is first expanded to \(B\times1\times2\times1000\). A temporal convolution with eight filters and a \(1\times125\) kernel learns frequency-selective responses. A depthwise \(2\times1\) spatial convolution applies two learned Fp1/Fp2 projections per temporal filter, producing 16 feature maps. Batch normalization, exponential linear units, average pooling, and dropout follow. A depthwise temporal convolution with a \(1\times31\) kernel and a pointwise \(1\times1\) convolution form the separable block. Global average pooling yields a 16-dimensional representation, and a linear layer produces three logits.

![Figure 2. FrontEEGNet computation and target-adaptation scopes.](figures/figure2_fronteegnet.svg)

**Figure 2. FrontEEGNet computation and target-adaptation scopes.** (a) The common signal-preparation pipeline produces a standardized 2-by-1000 input. (b) The compact network combines temporal, depthwise spatial, and separable convolutions before global average pooling and three-class prediction. (c) Blue cells indicate updated modules under each target-adaptation scope; grey cells remain frozen. Batch-normalization statistics remain fixed for head-only and partial adaptation.

**Table 1. FrontEEGNet architecture for the three-class COG-BCI experiment.**

| Stage | Operation | Output channels | Kernel / setting |
|---|---|---:|---|
| Input | Fp1/Fp2 waveform | 1 | \(2\times1000\) |
| Temporal filtering | Conv2D + batch normalization | 8 | \(1\times125\) |
| Spatial filtering | Depthwise Conv2D + batch normalization + ELU | 16 | \(2\times1\), depth multiplier 2 |
| Temporal reduction | Average pooling + dropout | 16 | pool \(1\times4\), dropout 0.5 |
| Separable filtering | Depthwise Conv2D + pointwise Conv2D + batch normalization + ELU | 16 | \(1\times31\), then \(1\times1\) |
| Readout | Average pooling + dropout + global average pooling | 16 | pool \(1\times8\), dropout 0.5 |
| Classification | Linear | 3 | 16-to-3 |

The model contains 1,915 trainable parameters. The temporal kernels correspond to 0.50 s and 0.124 s at 250 Hz. We selected the compact architecture because the wearable problem is data- and domain-limited rather than compute-limited.

### 3.4 Source training and inference

A separate FrontEEGNet instance is trained in each outer fold. Cross-entropy is optimized with Adam at a learning rate of \(10^{-3}\), batch size 256, and random seed 12345. Training runs for at most 40 epochs and stops after eight epochs without improvement on source-participant session 3. The selected checkpoint is then evaluated on the complete target-participant session 3.

For physical-device inference, a final deployment checkpoint is trained with sessions 1 and 2 from all 26 public participants and validated on their session 3 recordings. This checkpoint uses the same architecture and preprocessing and is not included in the leave-one-subject-out statistics. Incoming device samples are reconstructed on a uniform 250 Hz time grid inside each task block, filtered from 1 to 40 Hz, divided into 4 s windows at a 2 s stride, standardized per channel and window, and passed through the frozen network. Device calibration data set only a robust signal-amplitude rejection threshold; they do not update model parameters.

### 3.5 Target-participant adaptation

To test whether a small labelled target set improves the source model, we draw nested, class-balanced subsets of \(k\in\{1,2,4,8,16\}\) windows per class from target session 1. Because the original windows overlap by 50%, candidate selection first retains every other within-class window so that selected calibration examples do not overlap. Three independently shuffled nested sequences are evaluated. A \(k\)-shot subset is contained in every larger subset within the same repeat.

Four training conditions use identical calibration indices. **Head-only** updates the 51 parameters of the final 16-to-3 classifier at learning rate \(10^{-3}\). **Partial** updates the separable temporal depthwise and pointwise convolutions plus the classifier, for 803 trainable parameters at \(3\times10^{-4}\); all batch-normalization statistics remain fixed. **Full** updates all 1,915 source-initialized parameters at \(10^{-4}\). These three conditions run for 200 epochs. **Target-only scratch** initializes all parameters randomly and trains for 500 epochs at \(10^{-3}\), providing a convergence-strengthened non-transfer reference. No target validation set is used to select an epoch, and target session 3 remains untouched until final scoring. Repeated scores are first averaged within each participant before population inference.

### 3.6 Baselines

All primary baselines use the same 26 target folds. The spectral baselines average Welch PSD in theta (4–8 Hz), alpha (8–13 Hz), low beta (13–20 Hz), and high beta (20–30 Hz), yielding eight Fp1/Fp2 features. We evaluate a 27-parameter linear classifier, a 639-parameter MLP, and a 1,858-parameter two-node GCN. Covariance baselines use 2-by-2 Oracle Approximating Shrinkage covariance matrices with Riemannian minimum-distance-to-mean or tangent-space logistic regression. Euclidean and Riemannian Alignment variants estimate their target transform from unlabeled target session 1; this target access is reported explicitly and gives them more target information than FrontEEGNet.

### 3.7 Evaluation

Balanced accuracy is the primary metric:

$$
\operatorname{BAcc}=\frac{1}{C}\sum_{c=1}^{C}
\frac{\operatorname{TP}_{c}}{\operatorname{TP}_{c}+\operatorname{FN}_{c}}.
$$

Macro-F1 is the unweighted mean of class-wise F1. The participant, not the overlapping window, is the statistical unit, consistent with participant-level BCI benchmarking principles [10]. We report the mean and between-participant standard deviation. Confidence intervals are non-parametric bootstrap intervals for the mean using 10,000 participant-level resamples and seed 12345. Method comparisons bootstrap paired participant-level differences. Ablations and method comparisons are exploratory and are not corrected for multiplicity.

## 4 Experiments

### 4.1 Datasets and protocol

**COG-BCI MATB.** COG-BCI includes 29 participants recorded over three sessions separated by approximately one week [3]. A raw-archive audit found that every MATB session of sub-27 is file-identical to sub-17. The S1 MATB files of sub-25 and sub-28 are also identical, although their S2/S3 files differ, leaving participant identity ambiguous. We exclude sub-25, sub-27, and sub-28 before constructing any fold, leaving 26 participants. Every session contains Easy, Medium, and Difficult MATB blocks. After preprocessing, each participant contributes 148 windows per class per session, for 34,632 analysed windows. In each of 26 folds, sessions 1 and 2 from 25 source participants provide 22,200 training windows, their session 3 provides 11,100 validation windows, and the held-out participant's session 3 provides 444 test windows.

**OpenNeuro ds007169.** This dataset contains four n-back difficulty levels for 18 participants [14]. We retain Fp1/Fp2 and classify 1-, 2-, 3-, and 4-back. Within each source participant and class, the early approximately 75% of the recording is used for training and the late 25% for validation, separated by a one-stride guard. FrontEEGNet and PSD-MLP exclude the same prefix of each target recording and reserve all later windows for testing. FrontEEGNet does not use the excluded prefix, whereas PSD-MLP uses only its unlabeled features to estimate min–max normalization parameters and therefore receives more target-domain information. This is participant-disjoint retraining of the architecture, not direct transfer of COG-BCI weights to a new task.

**Device session.** One author self-experiment used the commercial Fp1/Fp2 device and an English-language lightweight MATB interface. Easy, Medium, and Difficult calibration blocks were completed. The test phase completed Difficult and then stopped partway through Easy because wearing discomfort became substantial. Medium testing and planned artifact blocks were not reached. The incomplete data are retained to test the acquisition and frozen-inference path and to characterize failure, not to estimate the planned three-class endpoint.

### 4.2 Main cross-subject comparison

FrontEEGNet achieves the highest mean performance among the evaluated methods (Table 2). Its balanced accuracy is 54.72% ± 5.46%, with macro-F1 of 52.70% ± 6.11%. The same-fold PSD-MLP reaches 49.15% ± 7.52%. The paired improvement is 5.57 percentage points with a 95% confidence interval of 1.91 to 9.34; FrontEEGNet wins on 18 participants, ties on one, and loses on seven. The interval excludes zero and all methods share the same target-participant folds; under these fixed implementations, the raw-waveform model has a higher participant-level mean than the PSD-MLP.

**Table 2. Strict participant-held-out performance on COG-BCI MATB. Values are mean ± between-participant standard deviation (%).**

| Method | Representation and target access | Parameters | Balanced accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Linear | Eight PSD values; target-S1 normalization | 27 | 46.65 ± 8.00 | — |
| PSD MLP | Eight PSD values; target-S1 normalization | 639 | 49.15 ± 7.52 | 49.04 ± 7.35 |
| PSD GCN | Two-node, four-band graph; target-S1 normalization | 1,858 | 49.06 ± 7.95 | 46.63 ± 10.28 |
| Tangent-space LR | 2-by-2 covariance; no target S1 | — | 44.53 ± 6.18 | 39.22 ± 7.54 |
| Riemannian MDM | 2-by-2 covariance; no target S1 | — | 43.49 ± 6.27 | 37.63 ± 9.15 |
| EA + tangent LR | Covariance; unlabeled target S1 | — | 43.99 ± 6.94 | 39.27 ± 8.71 |
| RA + tangent LR | Covariance; unlabeled target S1 | — | 45.17 ± 6.29 | 40.99 ± 7.80 |
| **FrontEEGNet** | Raw \(2\times1000\); no target S1/S2 | **1,915** | **54.72 ± 5.46** | **52.70 ± 6.11** |

The comparison narrows, but does not isolate, the explanation. The two-node GCN and the smaller PSD-MLP have similar means, while the covariance methods are weaker in these implementations. This pattern is consistent with four-band averaging discarding useful within-window information; however, representation, classifier, and optimization all differ, so the comparison cannot attribute the gain specifically to temporal patterns.

Class-wise recall is 75.03% for Easy, 36.46% for Medium, and 52.68% for Difficult. The model most reliably separates the lowest workload condition and struggles with the intermediate class. Its balanced accuracy therefore should not be interpreted as uniform three-class performance or as recovery of a continuous workload axis.

### 4.3 Low-shot target adaptation

Source initialization helps when the alternative is learning only from the target subset, but fine-tuning does not improve the source-model population mean (Table 3). With one label per class, head-only calibration reaches 51.06%, compared with 41.54% for target-only scratch. The paired head-minus-scratch gain is 9.52 points (95% CI: 7.17–11.77; 25 wins and one loss). Source-initialized methods outperform target-only scratch at every tested budget. This supports a matched-budget performance benefit, but it does not estimate how many labels are required to reach a prespecified performance target.

The stronger comparison is against zero-calibration inference. Source-only FrontEEGNet remains at 54.72%. Head-only adaptation approaches 53.31% with 16 labels per class; its paired difference of −1.41 points has a 95% interval of −3.27 to 0.32 and therefore includes zero. At smaller budgets, the head-only paired intervals are below zero, while partial and full fine-tuning show larger low-budget decreases; at 16 labels per class, partial and full reach 52.64% and 52.82%. Source initialization thus has a matched-budget advantage over target-only learning, but no tested update raises the source-only mean. The term *negative transfer* applies only to individual settings whose paired interval lies below zero, not to every budget.

The ordering across budgets and update scopes provides diagnostic, but not causal, evidence for this loss. Head-only performance approaches source-only as the calibration set grows, while updating 803 or 1,915 parameters is more harmful at the smallest budgets. This pattern is consistent with high-variance fitting and partial forgetting of population-level features. In addition, adaptation uses target session 1 whereas evaluation uses target session 3, so the update can absorb session-specific variation that does not persist to the test session. The experiment does not factorially isolate these mechanisms; they are treated as supported interpretations rather than established causes.

**Table 3. Nested target-S1 adaptation of the same FrontEEGNet. Values are balanced accuracy (%), averaged over three repeats within each participant and then across 26 participants.**

| Method | Trainable parameters | 1 label/class | 4 labels/class | 16 labels/class |
|---|---:|---:|---:|---:|
| Source-only | 0 | **54.72 ± 5.46** | **54.72 ± 5.46** | **54.72 ± 5.46** |
| Head-only | 51 | 51.06 ± 6.29 | 52.07 ± 5.21 | 53.31 ± 6.28 |
| Partial last block | 803 | 48.94 ± 5.98 | 50.54 ± 5.38 | 52.64 ± 6.43 |
| Full fine-tuning | 1,915 | 49.15 ± 6.15 | 51.54 ± 6.86 | 52.82 ± 7.28 |
| Target-only scratch | 1,915 | 41.54 ± 7.44 | 44.35 ± 7.94 | 47.66 ± 9.77 |

The complete 1/2/4/8/16-shot curve is retained in the machine-readable result. For deployment, these findings support source-only FrontEEGNet as the default; target updating should require independent evidence that the new session improves rather than assuming that personalization is beneficial.

### 4.4 Architecture ablation

We next isolate which choices are supported by the primary data (Table 4). Replacing the 125- and 31-sample temporal kernels with 63- and 15-sample kernels reduces mean balanced accuracy by 0.56 points, but the paired confidence interval includes zero. Replacing the learned two-channel spatial convolution with fixed common and difference components changes performance by only −0.01 points. Neither result supports describing long kernels or learned spatial weights as a new performance-critical module.

In contrast, removing either electrode produces a reliable reduction. Fp1-only performance is 50.85% and Fp2-only performance is 51.51%, corresponding to decreases of 3.87 and 3.21 points. Both paired intervals lie below zero. The model benefits from observing both frontal channels even though a fixed common/difference basis is sufficient to match the learned projection on average.

**Table 4. FrontEEGNet ablations on the same 26 folds.**

| Variant | Parameters | Balanced accuracy (%) | Difference from full (points) | Paired 95% CI |
|---|---:|---:|---:|---:|
| Full FrontEEGNet | 1,915 | 54.72 ± 5.46 | 0.00 | — |
| Short temporal kernels | 1,163 | 54.16 ± 5.77 | −0.56 | [−1.44, 0.28] |
| Fixed common/difference projection | 1,883 | 54.71 ± 5.56 | −0.01 | [−0.94, 0.91] |
| Fp1 only | 1,899 | 50.85 ± 7.02 | −3.87 | [−6.54, −1.01] |
| Fp2 only | 1,899 | 51.51 ± 7.52 | −3.21 | [−6.44, −0.06] |

### 4.5 External participant-disjoint validation

On ds007169, the four-class FrontEEGNet has 1,932 parameters because the output layer contains four units. Across 18 held-out participants, it obtains 30.16% ± 5.60% balanced accuracy and 23.60% ± 5.87% macro-F1. Its mean exceeds the 25% chance level by 5.16 points; the participant-bootstrap 95% interval for this excess is 2.57 to 7.76, and 14 of 18 participants are above chance.

The matched PSD-MLP reaches 29.77% ± 5.26%. FrontEEGNet's paired difference is only 0.39 points, with a 95% interval from −2.77 to 3.40. The above-chance mean and interval on the second dataset suggest participant-disjoint predictive information in the two-channel input, but they do not replicate FrontEEGNet's advantage over PSD on MATB. Pooled recalls of 55.03%, 13.92%, 44.13%, and 10.18% for 1- through 4-back further show severe confusion in the even-numbered classes. Dataset-level task and acquisition differences remain consequential.

### 4.6 Artifact robustness and physical-device diagnostic

**Synthetic artifact stress.**

Fp1/Fp2 are particularly exposed to frontal artifacts, so we add deterministic blink-like pulses and motion-like transients to every held-out MATB test fold without retraining. Clean balanced accuracy is 54.72%. Blink amplitudes of 50, 100, and 200 microvolts reduce it to 49.93%, 45.45%, and 40.38%, corresponding to losses of 4.79, 9.27, and 14.34 points. Motion amplitudes of 25, 50, and 100 microvolts yield 52.88%, 48.72%, and 41.60%, corresponding to losses of 1.84, 6.00, and 13.12 points. Every paired decrease has a participant-level 95% interval below zero.

These tests establish controlled sensitivity, not the prevalence of real artifacts. A synthetic waveform cannot determine whether clean-condition predictions arise from brain activity or naturally occurring eye and muscle correlates. Nevertheless, the monotonic degradation shows that a two-channel deployment needs signal-quality monitoring or artifact-aware training rather than relying on per-window standardization alone.

**Physical-device diagnostic.** The self-experiment records 276,810 two-channel samples at a nominal 250 Hz over 18.49 min. All three calibration blocks are complete. The independent test contains 149 windows from a complete 300 s Difficult block and 107 windows from 216.55 s of an interrupted Easy block; no Medium test block is available. The session stops because the rigid mounting structure provides limited adjustment, electrode contact is inconsistent, and local pressure becomes painful. These observations motivate compliant materials, adjustable support, electrode mounts that follow head curvature, improved pressure distribution, and continuous contact-quality feedback in future hardware.

For this diagnostic, the frozen deployment model is trained on all 26 public participants and is not updated with device labels. The device calibration segment sets only an amplitude-quality threshold. Five of 256 test windows are rejected by that rule. Across all recorded test windows, Difficult recall is 60.40% and Easy recall is 51.40%; predictions are distributed as 102 Easy, 28 Medium, and 126 Difficult. These two recalls are descriptive block diagnostics. We do not report their average as a primary balanced accuracy because the Easy block is incomplete and the Medium block is absent.

The acquisition audit also finds 106 sampling intervals longer than 8 ms and a maximum gap of 106.78 ms after reconstructing a monotonic received-order timeline. Physical units are not declared by the LSL stream and therefore cannot be independently verified from metadata. The experiment demonstrates that the software can acquire, store, reconstruct, preprocess, and classify physical-device data, but it also reveals a hardware-domain and ergonomics boundary that controlled public-data accuracy cannot resolve.

## 5 Discussion

Under the fixed primary protocol, a compact raw-waveform model shows higher participant-independent, cross-session predictive performance than the compared baselines after laboratory recordings are reduced to Fp1/Fp2. FrontEEGNet improves balanced accuracy by 5.57 points over the strongest same-fold PSD baseline while using no target-participant EEG before inference. This is a practically relevant comparison because the spectral MLP, GCN, covariance models, and alignment methods all operate under the same target folds, with target-data access disclosed.

The contribution is principally empirical and translational. EEGNet supplies the architectural foundation. Our work identifies a two-channel configuration, establishes a strict evaluation protocol, distinguishes supported design choices through ablation, and follows the model into an external dataset and physical acquisition path. The ablation results are important to this positioning: they do not justify claiming that long kernels or learned spatial filtering constitute a new mechanism. They do show that using both channels is more reliable than either alone.

The low-shot experiment separates two claims that are otherwise easy to conflate. Across the tested budgets, every source-initialized adaptation strategy outperforms target-only scratch. However, using target labels does not automatically improve the trained source model: the best adapted mean remains below source-only, and updating more parameters is particularly harmful at the smallest budgets. Among the candidate policies covered by this protocol, zero calibration therefore has the highest mean rather than being an unevaluated default.

Several non-exclusive mechanisms may explain the negative transfer. The source model is estimated from 25 participants and selected on held-out source sessions, whereas each target update uses only 3 to 48 labelled windows from a single session. Such a small adaptation set can give a few atypical or artifact-contaminated windows disproportionate influence and can move the representation away from a population-stable solution. The monotonic recovery of head-only adaptation with increasing label budget and the larger low-budget loss from broader updates are consistent with this explanation. Cross-session non-stationarity provides a second explanation because calibration and testing occur in target sessions 1 and 3, respectively. Full fine-tuning may additionally be affected by unreliable batch-normalization updates, although partial adaptation keeps batch-normalization statistics fixed and still degrades, so this factor cannot explain the result alone. Finally, the leakage-free fixed-epoch protocol does not use target-session validation and may miss a participant-specific early-stopping point. Distinguishing these alternatives requires an independent target validation segment and a later untouched test session; the present experiment establishes the performance boundary, not a unique causal mechanism.

The external result tempers the primary gain. FrontEEGNet remains above chance on four-class n-back data, but its advantage over PSD is uncertain and its class-wise errors are severe. This may reflect the different task, label semantics, acquisition system, preprocessing, or participant population. It also illustrates why a model selected on one EEG benchmark should not be presented as universally transferable.

The artifact and device evidence identify two deployment bottlenecks. First, frontal transient corruption can erase a large part of the controlled-data advantage. Because no dedicated EOG or motion annotation is present in the model input, the present design cannot determine whether predictions arise from neural activity or from task-correlated ocular, muscular, or mechanical signals. Second, nominal channel equivalence does not guarantee cross-hardware equivalence. Unknown units, timing gaps, electrode contact, and reference differences can move the signal outside the training domain even when both systems expose Fp1 and Fp2.

Several limitations remain. COG-BCI provides the only repeated-session benchmark in the main evaluation. The FrontEEGNet family and current configuration were selected after inspecting comparisons on the same 26 folds, without an independent outer dataset confirming that choice; the resulting benchmark evidence is therefore exploratory. The 26-fold result uses a single predefined optimization seed and does not fully quantify training stochasticity. Overlapping 4 s windows are correlated, although participant-level inference avoids treating them as independent samples. Task conditions are proxies for workload rather than window-level ground truth of a latent cognitive state. Synthetic artifacts do not reproduce the complete morphology or occurrence statistics of real blinks and movement. The external dataset retrains the architecture and therefore evaluates architectural replication, not direct transfer of learned weights. Finally, the physical-device session includes one author participant and ends before all test classes are complete; it cannot establish device efficacy or population generalization.

Future work should retain the strict subject-held-out design while adding complete cross-day wearable recordings, synchronized EOG or eye tracking, motion sensors, verified units, and contact-quality measurements. Artifact-aware augmentation can then be trained and assessed against recorded events rather than synthetic waveforms alone. Cross-device adaptation should be evaluated with an untouched post-adaptation test session. The present results suggest that improving measurement stability and domain handling is likely to be more valuable than enlarging the 1,915-parameter classifier.

## 6 Conclusion

We evaluated, rather than invented, an EEGNet configuration for two-channel wearable workload decoding. FrontEEGNet reaches 54.72% balanced accuracy under a participant- and session-disjoint COG-BCI MATB protocol, with a paired interval against the same-fold PSD-MLP that excludes zero. Source-initialized methods outperform target-only scratch at matched label budgets, but no tested low-shot update raises the zero-calibration population mean. Both Fp1 and Fp2 provide information, whereas the present evidence does not support treating the selected temporal kernels or learned spatial projection as independent innovations. The ds007169 mean is above chance but does not establish an advantage over PSD; synthetic corruption and one interrupted self-experiment likewise cannot establish wearable efficacy. Overall, the study supports two-channel participant-independent decoding only under the specified controlled protocol. Cross-task, cross-hardware, and real-world robustness require further validation.

## References

[1] Zander, T. O. and Kothe, C. (2011). Towards passive brain–computer interfaces: applying brain–computer interface technology to human–machine systems in general. *Journal of Neural Engineering*, 8, 025005. https://doi.org/10.1088/1741-2560/8/2/025005

[2] Debener, S., Minow, F., Emkes, R., Gandras, K., and de Vos, M. (2012). How about taking a low-cost, small, and wireless EEG for a walk? *Psychophysiology*, 49, 1617–1621. https://doi.org/10.1111/j.1469-8986.2012.01471.x

[3] Hinss, M. F., Jahanpour, E. S., Somon, B., Pluchon, L., Dehais, F., and Roy, R. N. (2023). Open multi-session and multi-task EEG cognitive dataset for passive brain-computer interface applications. *Scientific Data*, 10, 85. https://doi.org/10.1038/s41597-022-01898-y

[4] Chikhi, S., Matton, N., and Blanchet, S. (2022). EEG power spectral measures of cognitive workload: a meta-analysis. *Psychophysiology*, 59, e14009. https://doi.org/10.1111/psyp.14009

[5] Wu, D., Xu, Y., and Lu, B.-L. (2022). Transfer learning for EEG-based brain–computer interfaces: a review of progress made since 2016. *IEEE Transactions on Cognitive and Developmental Systems*, 14, 4–19. https://doi.org/10.1109/TCDS.2020.3007453

[6] He, H. and Wu, D. (2020). Transfer learning for brain–computer interfaces: a Euclidean space data alignment approach. *IEEE Transactions on Biomedical Engineering*, 67, 399–410. https://doi.org/10.1109/TBME.2019.2913914

[7] Barachant, A., Bonnet, S., Congedo, M., and Jutten, C. (2012). Multiclass brain–computer interface classification by Riemannian geometry. *IEEE Transactions on Biomedical Engineering*, 59, 920–928. https://doi.org/10.1109/TBME.2011.2172210

[8] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., and Lance, B. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15, 056013. https://doi.org/10.1088/1741-2552/aace8c

[9] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W., and Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38, 5391–5420. https://doi.org/10.1002/hbm.23730

[10] Jayaram, V. and Barachant, A. (2018). MOABB: trustworthy algorithm benchmarking for BCIs. *Journal of Neural Engineering*, 15, 066011. https://doi.org/10.1088/1741-2552/aadea0

[11] Demirezen, G., Brouwer, A.-M., and Taşkaya Temizel, T. (2026). Optimal transport and graph neural networks for cross-session mental workload classification. *Applied Sciences*, 16, 5506. https://doi.org/10.3390/app16115506

[12] Charles, R. L. and Nixon, J. (2019). A systematic review of physiological measures of mental workload. *Applied Ergonomics*, 74, 221–232. https://doi.org/10.1016/j.apergo.2018.08.028

[13] Croft, R. J. and Barry, R. J. (2000). Removal of ocular artifact from the EEG: a review. *Neurophysiologie Clinique*, 30, 5–19. https://doi.org/10.1016/S0987-7053(00)00055-1

[14] Booth, L. and Barras, M. (2026). Multimodal Cognitive Workload n-back Task, 4 Difficulties (version 1.0.5). OpenNeuro dataset. https://doi.org/10.18112/openneuro.ds007169.v1.0.5
