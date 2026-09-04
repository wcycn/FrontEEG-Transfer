# FrontEEG-Transfer

**An empirical study of participant-independent mental-workload decoding using only Fp1/Fp2 EEG.**

FrontEEG-Transfer 面向双通道前额 EEG 设备，从公开多被试数据学习可迁移的心理负荷模型，
并在新使用者上比较零校准推理、少标签微调和从零训练。当前任务为
`Easy / Medium / Difficult` 三分类。唯一主模型 FrontEEGNet 是适配双通道输入的 EEGNet
实现，共 1,915 个参数；本项目不声称发明 EEGNet。当前证据主要来自公开数据上的
探索性比较，不应解读为已验证的注意力测量、临床系统或可穿戴产品。

## Method

```text
Fp1/Fp2 EEG
  -> 1–40 Hz band-pass and 250 Hz resampling
  -> 4 s windows with a 2 s stride
  -> per-window, per-channel z-score
  -> temporal convolution
  -> depthwise two-channel spatial convolution
  -> separable temporal convolution
  -> global average pooling and a 16-to-3 classifier
  -> Easy / Medium / Difficult
```

每个窗口以 `2 × 1000` 原始波形直接输入网络。FrontEEGNet 使用紧凑时间卷积、跨 Fp1/Fp2
的 depthwise 空间卷积和 separable temporal block 提取特征，随后通过全局平均池化完成
三分类。Source-only 推理不估计目标参与者统计量，也不需要目标标签。

## Evaluation protocol

公开数据实验使用 COG-BCI 的 MATB-II `Easy / Medium / Difficult` 任务。原始档案审计显示，
`sub-27` 的三个 MATB 会话与 `sub-17` 逐文件相同；`sub-25` 与 `sub-28` 的 S1 MATB 记录
完全相同，但 S2/S3 不同，无法唯一确定受试者身份。因此排除这 3 名参与者，最终评估 26 人。
参与者和会话互斥的 leave-one-subject-out 协议为：

- 其余 25 名 source 参与者的 S1 和 S2：监督训练；
- 其余 25 名 source 参与者的 S3：学习率选择与早停；
- held-out 目标参与者的 S1：仅在显式微调实验中抽取少标签子集；
- held-out 目标参与者的 S3：最终测试；
- held-out 目标参与者的 S2：不使用，S3 不参与训练或模型选择。

校准预算按类别嵌套抽取，较小预算的样本始终包含在较大预算中。实验比较：

- `source-only`：不使用目标标签；
- `target scratch`：仅使用目标少量标签，从随机初始化开始训练；
- `head calibration`：加载 source 权重，只更新分类头；
- `partial fine-tuning`：更新最后一个可分离时间卷积块和分类头；
- `full fine-tuning`：加载 source 权重，微调整个 FrontEEGNet。

## Results

下表为 26 名 held-out 目标参与者上的平均 balanced accuracy。Scratch 训练 500 epoch，
三种 source 初始化微调训练 200 epoch；每个预算进行 3 次受控嵌套重复。

| Labels per class | Source-only | Head-only | Partial | Full | Target scratch |
|---:|---:|---:|---:|---:|---:|
| 1 | **54.72%** | 51.06% | 48.94% | 49.15% | 41.54% |
| 4 | **54.72%** | 52.07% | 50.54% | 51.54% | 44.35% |
| 16 | **54.72%** | 53.31% | 52.64% | 52.82% | 47.66% |

FrontEEGNet 的 source-only balanced accuracy 为 **54.72% ± 5.46%**，三分类机会水平为
33.33%。多被试监督初始化在所有预算下均明显优于相同目标标签从零训练；例如 Head-only
1-shot 比 scratch 高 9.52 个百分点（95% CI：7.17–11.77）。但是三种微调均未使总体均值
超过 source-only。最接近的是 Head-only 16-shot，仍低 1.41 个百分点，其 95% CI 为
−3.27–0.32。因此零校准不是遗漏微调，而是当前实验支持的部署选择。

在同一 LOSO 划分下，FrontEEGNet 高于 PSD-MLP 5.57 个百分点（配对受试者 bootstrap
95% CI：1.91–9.34），也高于所测试的协方差、对齐、Linear 和 GCN 基线。结构消融显示，
Fp1-only 和 Fp2-only 分别比完整双通道低 3.87 和 3.21 个百分点；较短时间卷积核与固定
共模/差模空间投影则未显示稳定差异。

相同架构在 OpenNeuro ds007169 四分类严格 18-fold LOSO 上重新训练，达到
**30.16% ± 5.60%**（机会水平 25%），但相对 PSD 的优势不显著。合成眨眼和运动瞬态会随
污染幅度增加而系统性降低性能；这些结果刻画模型脆弱性，不等同于真人伪迹验证。完整统计见
[`docs/matb_fronteegnet_lowshot_adaptation.md`](docs/matb_fronteegnet_lowshot_adaptation.md)、
[`docs/matb_fronteegnet_architecture_ablation.md`](docs/matb_fronteegnet_architecture_ablation.md)、
[`docs/ds007169_eegnet_external_validation.md`](docs/ds007169_eegnet_external_validation.md) 和
[`docs/artifact_stress_results.md`](docs/artifact_stress_results.md)。

## Installation

```bash
conda create -n fronteeg python=3.11 -y
conda activate fronteeg
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

如果需要指定 CUDA 版本，请先按照 PyTorch 官方说明安装与本机驱动匹配的 PyTorch；随后
执行上述命令时，已满足版本约束的 PyTorch 不会被重复替换。

## Data preparation

COG-BCI 数据来自 [Zenodo record 7413650](https://zenodo.org/records/7413650)。下载约
31.7 GB，请预留足够空间：

```bash
bash scripts/download_cog_bci.sh data/cog_bci_raw

python scripts/prepare_cog_matb_paper_all.py \
  --archive-directory data/cog_bci_raw \
  --output-directory data/cog_matb_paper \
  --jobs 4
```

数据、训练输出、日志和私人实机会话均由 `.gitignore` 排除。

## Reproduce the cross-subject experiment

先准备双通道原始波形，再运行 26 折 FrontEEGNet source-only 实验：

```bash
python scripts/prepare_cog_matb_raw_all.py \
  --archive-directory data/cog_bci_raw \
  --output-directory data/cog_matb_raw_fp1_fp2 \
  --jobs 4

python scripts/run_matb_eegnet_all.py \
  --raw-root data/cog_matb_raw_fp1_fp2 \
  --output-directory outputs/eegnet_loso \
  --gpus 0 1
```

在完全相同的 source checkpoint 和目标样本索引上运行 Head-only、Partial、Full 与 Scratch：

```bash
python scripts/run_matb_fronteegnet_adaptation_all.py \
  --raw-root data/cog_matb_raw_fp1_fp2 \
  --source-directory outputs/eegnet_loso \
  --output-directory outputs/fronteegnet_adaptation \
  --gpus 0 1 \
  --budgets 1 2 4 8 16 \
  --repeats 3

python scripts/summarize_matb_fronteegnet_adaptation.py \
  --input-directory outputs/fronteegnet_adaptation/subjects \
  --output-json outputs/fronteegnet_adaptation/summary.json \
  --output-markdown outputs/fronteegnet_adaptation/summary.md
```

第二数据集的输入是从 OpenNeuro ds007169 生成的 Fp1/Fp2 窗口文件，命名为
`sub-XX_nback-{1,2,3,4}_test.npz`；每名受试者、每个难度各一个文件，需包含形状为
`[N, 2, 800]` 的 200 Hz `eeg` 和对应 `label`。当前公开代码从这一预处理窗口表示开始；
原始数据下载、许可证与访问规则以 OpenNeuro 为准。运行 FrontEEGNet 与匹配 PSD-MLP 基线：

```bash
python scripts/run_ds007169_eegnet_all.py \
  --raw-root data/ds007169_fp1_fp2 \
  --output-directory outputs/ds007169_eegnet \
  --gpus 0 1

python scripts/prepare_ds007169_psd.py \
  --input-directory data/ds007169_fp1_fp2 \
  --output-directory data/ds007169_psd

python scripts/run_ds007169_all.py \
  --feature-root data/ds007169_psd \
  --output-directory outputs/ds007169_psd \
  --gpus 0 1

python scripts/summarize_ds007169.py \
  --input-directory outputs/ds007169_psd/subjects \
  --output-json outputs/ds007169_psd/summary.json \
  --output-markdown outputs/ds007169_psd/summary.md

python scripts/summarize_ds007169_eegnet.py \
  --input-directory outputs/ds007169_eegnet/subjects \
  --psd-reference outputs/ds007169_psd/summary.json \
  --output-json outputs/ds007169_eegnet/summary.json \
  --output-markdown outputs/ds007169_eegnet/summary.md
```

重新训练用于设备部署的全 source FrontEEGNet：

```bash
python scripts/train_final_fronteegnet_source.py \
  --raw-root data/cog_matb_raw_fp1_fp2 \
  --checkpoint checkpoints/final_fronteegnet_source.pt \
  --output-json results/final_fronteegnet_source_training.json \
  --device cuda:0
```

仓库已经提供部署权重：`checkpoints/final_fronteegnet_source.pt`。

## Real-device experiment

独立实机客户端维护在
[`device-experiment`](https://github.com/wcycn/FrontEEG-Transfer/tree/device-experiment)
分支。该分支包含当前 MATB-lite PsychoPy 范式、LSL 检查、原始 EEG 增量保存、会话验收和
Windows/Linux 配置说明；`main` 分支不再提供另一套实机操作流程。

只下载实机客户端：

```bash
git clone --branch device-experiment --single-branch --depth 1 \
  https://github.com/wcycn/FrontEEG-Transfer.git
```

## Paper and presentation

- [中文报告](paper/manuscript_zh.md)
- [English manuscript](paper/manuscript_en.md)
- [Figures](paper/figures)
- [5-minute presentation](paper/presentation/FrontEEGNet_5min_Chinese_Presentation.pptx)

## Repository layout

```text
checkpoints/final_fronteegnet_source.pt deployment checkpoint
paper/manuscript_zh.md                  Chinese report
paper/manuscript_en.md                  English manuscript
paper/figures/                          publication figures
paper/presentation/                     five-minute presentation
src/fronteeg_transfer/baselines.py      FrontEEGNet and comparison models
scripts/prepare_cog_matb_raw_all.py      raw Fp1/Fp2 preprocessing
scripts/run_matb_eegnet_all.py           strict LOSO source training
scripts/run_matb_fronteegnet_adaptation_all.py
                                         low-shot adaptation comparison
scripts/train_final_fronteegnet_source.py final all-source model training
```

## Tests

```bash
ruff check src scripts tests
pytest -q
```

测试覆盖协议划分、嵌套少标签抽样、PSD 与原始波形特征、模型、外部数据、合成伪迹和训练
组件。测试数量会随实现更新，因此 README 不固定声明用例总数。

## License

本项目原创代码、文档和图表采用 [MIT License](LICENSE)。改编的第三方实现分别保留其
原始许可证和来源声明，见 [GNN-OT notice](GNN_OT_NOTICE.md) 与
[mindscape notice](MINDSCAPE_NOTICE.md)。COG-BCI 数据不随仓库分发；其官方 Zenodo
版本采用 [CC BY 4.0](https://zenodo.org/records/7413650)。

## Limitations

- 主模型与配置是在同一组 26 折比较后选定的，因此属于探索性证据，不是独立预注册验证。
- 公开数据使用 TP10 参考；不同设备参考方式会产生域偏移。
- 只使用 Fp1/Fp2 无法获得全头皮 EEG 的空间信息。
- 相邻窗口有 50% 重叠，统计推断必须以参与者为单位，不能把窗口视为独立样本。
- 在线滤波与训练预处理的小差异可能改变边界样本的分类结果。
- 公开数据结果不能替代实体设备上的独立 calibration/test 验证。
