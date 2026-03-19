# ASR Project — Architecture & Developer Guide

> 本文档描述项目的完整架构，供后续开发（新增模型、MLC 集成等）使用。

---

## 目录

1. [项目概览](#1-项目概览)
2. [目录结构](#2-目录结构)
3. [配置系统](#3-配置系统)
4. [模型架构](#4-模型架构)
5. [编码器（Encoders）](#5-编码器encoders)
6. [解码器（Decoders）](#6-解码器decoders)
7. [核心模块（Modules）](#7-核心模块modules)
8. [损失函数（Criterion）](#8-损失函数criterion)
9. [解码策略（Search）](#9-解码策略search)
10. [训练框架](#10-训练框架)
11. [数据与特征提取](#11-数据与特征提取)
12. [Tokenizer](#12-tokenizer)
13. [语言模型（LM）](#13-语言模型lm)
14. [优化器与调度器](#14-优化器与调度器)
15. [评估指标](#15-评估指标)
16. [依赖库](#16-依赖库)
17. [新增模型指南](#17-新增模型指南)
18. [MLC 集成指南](#18-mlc-集成指南)

---

## 1. 项目概览

这是一个基于 **PyTorch + PyTorch Lightning** 的模块化 ASR（自动语音识别）框架。支持多种主流 ASR 架构，配置系统基于 **Hydra + OmegaConf**，支持灵活组合训练配置。

**已实现的模型架构：**

| 模型 | 编码器 | 解码器 | 损失 |
|------|--------|--------|------|
| Conformer + LSTM | ConformerEncoder | LSTMAttentionDecoder | CE / CTC+CE |
| Transformer | TransformerEncoder | TransformerDecoder | CE / CTC+CE |
| LAS (Listen, Attend, Spell) | LSTMEncoder | LSTMAttentionDecoder | CE / CTC+CE |
| LAS + Location Aware | LSTMEncoder | LocationAwareAttention | CE / CTC+CE |
| Transformer Transducer | TransformerTransducerEncoder | TransformerTransducerDecoder | RNN-T |
| Transformer LM | — | TransformerDecoder | CE (LM) |

---

## 2. 目录结构

```
asr/
├── asr/                                # 主包
│   ├── __init__.py                     # 顶层 import (criterion, models, etc.)
│   ├── callbacks.py                    # Lightning callbacks (CheckpointEveryNSteps)
│   ├── utils.py                        # 通用工具函数
│   │
│   ├── configs/                        # YAML 默认配置
│   │   ├── train.yaml
│   │   └── eval.yaml
│   │
│   ├── dataclass/                      # 所有配置 Dataclass
│   │   ├── configurations.py           # ASRDataclass 基类 + 所有子配置
│   │   └── initialize.py              # Hydra ConfigStore 注册入口
│   │
│   ├── criterion/                      # 损失函数
│   │   ├── ctc/ctc.py                  # CTCLoss
│   │   └── cross_entropy/cross_entropy.py  # CrossEntropyLoss
│   │
│   ├── data/                           # 数据处理
│   │   ├── audio/
│   │   │   ├── dataset.py              # SpeechToTextDataset
│   │   │   ├── load.py                 # 音频文件读取 (librosa)
│   │   │   ├── melspectrogram/         # 梅尔频谱特征
│   │   │   ├── filter_bank/            # 梅尔滤波器特征
│   │   │   ├── spectrogram/            # 原始 STFT 特征
│   │   │   └── mfcc/                   # MFCC 特征
│   │   └── text/                       # 文本处理工具
│   │
│   ├── datasets/                       # 数据集 + Lightning DataModule
│   │   ├── librispeech/lit_data_module.py
│   │   └── aishell/lit_data_module.py
│   │
│   ├── decoders/                       # 解码器
│   │   ├── asr_decoder.py              # 抽象基类
│   │   ├── lstm_attention_decoder.py
│   │   ├── transformer_decoder.py
│   │   ├── rnn_transducer_decoder.py
│   │   └── transformer_transducer_decoder.py
│   │
│   ├── encoders/                       # 编码器
│   │   ├── asr_encoder.py              # 抽象基类
│   │   ├── conformer_encoder.py
│   │   ├── lstm_encoder.py
│   │   ├── convolutional_lstm_encoder.py
│   │   └── transformer_transducer_encoder.py
│   │
│   ├── lm/                             # 语言模型
│   │   ├── asr_lm.py                   # 抽象基类
│   │   ├── lstm_lm.py
│   │   └── transformer_lm.py
│   │
│   ├── metrics/
│   │   └── metrics.py                  # WER, CER
│   │
│   ├── models/                         # 完整模型 (Lightning Modules)
│   │   ├── asr_model.py                # ASRModel 基类
│   │   ├── asr_encoder_decoder.py      # 编码器-解码器基类
│   │   ├── asr_ctc_model.py            # CTC 模型基类
│   │   ├── asr_transducer_model.py     # Transducer 模型基类
│   │   ├── conformer/models.py         # ConformerLSTMModel
│   │   ├── transformer/model.py        # TransformerModel 等
│   │   ├── listen_attend_spell/model.py
│   │   ├── transformer_transducer/model.py
│   │   └── transformer_lm/model.py
│   │
│   ├── modules/                        # 可复用网络模块
│   │   ├── wrapper.py                  # Linear, View, Transpose
│   │   ├── swish.py                    # Swish 激活
│   │   ├── glu.py                      # GLU
│   │   ├── mask.py                     # 注意力掩码生成
│   │   ├── positional_encoding.py      # 绝对/相对位置编码
│   │   ├── transformer_embedding.py    # Token + PE 嵌入
│   │   ├── residual_connection_module.py
│   │   ├── dot_product_attention.py
│   │   ├── additive_attention.py       # Bahdanau 注意力
│   │   ├── multi_head_attention.py
│   │   ├── relative_multi_head_attention.py
│   │   ├── location_aware_attention.py
│   │   ├── positionwise_feed_forward.py
│   │   ├── conformer_feed_forward_module.py
│   │   ├── conformer_attention_module.py
│   │   ├── conformer_convolution_module.py
│   │   ├── conformer_block.py
│   │   ├── conv2d_subsampling.py
│   │   ├── conv2d_extractor.py
│   │   ├── depthwise_conv1d.py
│   │   ├── depthwise_conv2d.py
│   │   ├── pointwise_conv1d.py
│   │   └── mask_conv2d.py
│   │
│   ├── optim/
│   │   ├── radam.py, adamp.py, novograd.py  # 自定义优化器
│   │   └── scheduler/                        # 学习率调度器
│   │       ├── lr_scheduler.py               # 基类
│   │       ├── transformer_lr_scheduler.py
│   │       ├── warmup_scheduler.py
│   │       ├── reduce_lr_on_plateau_scheduler.py
│   │       └── warmup_reduce_lr_on_plateau_scheduler.py
│   │
│   ├── search/                         # 解码搜索策略
│   │   ├── beam_search_base.py
│   │   ├── beam_search_ctc.py
│   │   ├── beam_search_lstm.py
│   │   ├── beam_search_rnn_transducer.py
│   │   └── beam_search_transformer_transducer.py
│   │
│   └── tokenizers/                     # 分词器
│       ├── tokenizer.py                # 抽象基类
│       └── librispeech/
│           ├── character.py            # 字符级
│           └── subword.py              # 子词级 (BPE/SentencePiece)
│
├── asr_cli/                            # 命令行接口
├── scripts/                            # 训练脚本
├── requirements.txt
└── train_librispeech.sh
```

---

## 3. 配置系统

配置基于 **Hydra + OmegaConf**，所有配置类继承自 `ASRDataclass`。

### 注册表结构

| 注册表 | 配置类 | 默认值 |
|--------|--------|--------|
| `AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY` | melspectrogram, mfcc, spectrogram, filter_bank | melspectrogram |
| `AUGMENT_DATACLASS_REGISTRY` | AugmentConfigs | — |
| `DATASET_DATACLASS_REGISTRY` | LibriSpeechConfigs | librispeech |
| `MODEL_DATACLASS_REGISTRY` | ConformerLSTMConfigs, TransformerConfigs, ... | conformer_lstm |
| `CRITERION_DATACLASS_REGISTRY` | CTCLossConfigs, CrossEntropyLossConfigs | cross_entropy |
| `SCHEDULER_DATACLASS_REGISTRY` | TransformerLRSchedulerConfigs, WarmupLRSchedulerConfigs, ... | transformer |
| `TOKENIZER_DATACLASS_REGISTRY` | LibriSpeechCharacterTokenizerConfigs | libri_character |
| `TRAINER_DATACLASS_REGISTRY` | GPUTrainerConfigs, CPUTrainerConfigs, ... | gpu |

### 新增模型配置步骤

```python
# 1. 在 dataclass/configurations.py 中定义配置类
@dataclass
class MyModelConfigs(ASRDataclass):
    model_name: str = field(default="my_model", metadata={"help": "..."})
    hidden_dim: int = field(default=512, metadata={"help": "..."})
    ...

# 2. 注册到 MODEL_DATACLASS_REGISTRY
MODEL_DATACLASS_REGISTRY["my_model"] = MyModelConfigs

# 3. 在 dataclass/initialize.py 的 hydra_train_init() 中注册到 ConfigStore
cs.store(group="model", name="my_model", node=MyModelConfigs)
```

---

## 4. 模型架构

### 4.1 ConformerLSTMModel

```
Audio (B, T, 80)
  └─► Conv2dSubSampling ──► (B, T/4, encoder_dim)
        └─► Linear + Dropout
              └─► ConformerBlock × 17
                    ├─ FeedForward (0.5x residual)
                    ├─ RelativeMultiHeadSelfAttention
                    ├─ ConformerConvModule
                    └─ FeedForward (0.5x residual)
                          └─► (optional) CTC head
                                └─► LSTMAttentionDecoder (auto-regressive)
                                      └─► logits (B, T', num_classes)
```

**关键参数** (`ConformerLSTMConfigs`):

```python
encoder_dim: int = 512
num_encoder_layers: int = 17
num_attention_heads: int = 8
feed_forward_expansion_factor: int = 4
conv_expansion_factor: int = 2
conv_kernel_size: int = 31
half_step_residual: bool = True
joint_ctc_attention: bool = True
# Decoder
num_decoder_layers: int = 2
decoder_hidden_state_dim: int = 512
decoder_attn_mechanism: str = 'multi-head'  # 'dot', 'additive', 'loc', 'multi-head'
max_length: int = 128
teacher_forcing_ratio: float = 1.0
```

### 4.2 TransformerModel

```
Audio (B, T, 80)
  └─► Feature Extractor (optional VGG / Conv2D subsampling)
        └─► TransformerEncoder × 12
              └─► TransformerDecoder × 6
                    └─► logits (B, T', num_classes)
```

**变体**: `TransformerModel`, `JointCTCTransformerModel`, `VGGTransformerModel`

**关键参数** (`TransformerConfigs`):
```python
d_model: int = 512
d_ff: int = 2048
num_attention_heads: int = 8
num_encoder_layers: int = 12
num_decoder_layers: int = 6
joint_ctc_attention: bool = False
ffnet_style: str = 'ff'  # 'ff' or 'conv'
```

### 4.3 Transformer Transducer

```
Audio (B, T, 80)
  └─► TransformerTransducerEncoder
        └─► encoder_output (B, T', encoder_dim)

Labels (B, U)
  └─► TransformerTransducerDecoder
        └─► decoder_output (B, U, decoder_dim)

Joint Network:
  _expand_for_joint(encoder_output, decoder_output)
  → (B, T', U, encoder_dim + decoder_dim)
  → FC → Tanh → FC → logits (B, T', U, num_classes)

Loss: RNN-T Loss (warp-rnnt)
Inference: BeamSearchTransformerTransducer
```

### 4.4 新增模型的继承关系

```
pl.LightningModule
  └─► ASRModel                    # configure_optimizers, configure_criterion, logging
        ├─► ASREncoderDecoderModel # collect_outputs, joint CTC+CE loss
        │     ├─► ConformerLSTMModel
        │     ├─► TransformerModel
        │     └─► ListenAttendSpellModel
        ├─► ASRCTCModel            # CTC-only loss
        └─► ASRTransducerModel     # joint network, RNN-T loss
```

---

## 5. 编码器（Encoders）

所有编码器继承自 `ASREncoder(nn.Module)`：

```python
class ASREncoder(nn.Module):
    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, ...]:
        raise NotImplementedError

    def count_parameters(self) -> int: ...
    def update_dropout(self, dropout_p: float) -> None: ...
```

### ConformerEncoder

- **输入**: `(B, T, input_dim)` + `input_lengths (B,)`
- **输出**: `(outputs, encoder_logits, output_lengths)`
  - `outputs`: `(B, T', encoder_dim)` — T' ≈ T/4（经过 Conv2dSubSampling）
  - `encoder_logits`: `(B, num_classes, T')` 或 `None`（joint_ctc_attention=False 时）
  - `output_lengths`: `(B,)` 下采样后的有效长度

### LSTMEncoder

- **输入**: `(B, T, input_dim)` + `input_lengths`
- **输出**: `(outputs, ctc_logits, input_lengths)`
- 支持双向（bidirectional），输出维度 = `hidden_state_dim * 2`

### TransformerTransducerEncoder

- Pre-layer normalization 架构
- 正弦位置编码（固定）
- 每层输出: `(output, attn_distribution)`

---

## 6. 解码器（Decoders）

所有解码器继承自 `ASRDecoder(nn.Module)`。

### LSTMAttentionDecoder

自回归 LSTM 解码器，支持 5 种注意力机制：

| `decoder_attn_mechanism` | 类 | 论文 |
|--------------------------|-----|------|
| `'dot'` | `DotProductAttention` | Luong et al. |
| `'additive'` | `AdditiveAttention` | Bahdanau et al. arXiv:1409.0473 |
| `'loc'` | `LocationAwareAttention` | Chorowski et al. arXiv:1506.07503 |
| `'multi-head'` | `MultiHeadAttention` | Vaswani et al. arXiv:1706.03762 |
| `'scaled-dot'` | `DotProductAttention(scaled=True)` | — |

**前向传播**:
- 训练: 以 `teacher_forcing_ratio` 决定是否用 ground truth token
- 推理: `teacher_forcing_ratio=0`，用上一步预测 token 作为输入

### TransformerDecoder / TransformerDecoderLayer

标准 Transformer 解码器，带 causal mask（`get_attn_subsequent_mask`）和 cross-attention。

### RNNTransducerDecoder / TransformerTransducerDecoder

Transducer 的预测网络（Prediction Network）：
- 只依赖已预测的 label 序列
- 不直接 attend 到编码器输出（联合由 Joint Network 完成）

---

## 7. 核心模块（Modules）

### 注意力机制

```python
# 所有注意力模块的统一接口
attention(query, key, value, mask=None) -> (context, attn_weights)
```

| 模块 | 特点 |
|------|------|
| `DotProductAttention` | Q·K^T / sqrt(d), softmax, ·V |
| `AdditiveAttention` | MLP(Q, K) → energy → softmax |
| `MultiHeadAttention` | 多头并行，concat + 线性映射 |
| `RelativeMultiHeadAttention` | 相对位置偏置（Transformer-XL 风格） |
| `LocationAwareAttention` | 在加性注意力基础上加入累积对齐向量的卷积特征 |

### 位置编码

```python
# 绝对位置编码（Transformer 原版）
PositionalEncoding(d_model, max_len=5000)
# 输入/输出: (B, T, d_model)

# 相对位置编码（Conformer / Transformer-XL）
RelPositionalEncoding(d_model, max_len=5000)
# 返回 (pos_embedding, pe) 供 RelativeMultiHeadAttention 使用
```

### Conformer 组件

```python
ConformerBlock(
    encoder_dim,          # 模型维度
    num_attention_heads,  # 注意力头数
    feed_forward_expansion_factor,  # FFN 扩展因子（默认 4）
    conv_expansion_factor,          # 卷积扩展因子（默认 2）
    conv_kernel_size,               # 深度卷积核大小（默认 31，必须为奇数）
    half_step_residual,             # FFN 残差是否缩放 0.5
    ...dropout params...
)
```

内部结构（Macaron 结构）：
```
x → FeedForward(0.5x) → RelMultiHeadSelfAttn → ConvModule → FeedForward(0.5x) → LayerNorm
```

### 卷积前端

```python
# 4× 下采样（两个 stride=2 卷积）
Conv2dSubSampling(input_dim, in_channels=1, out_channels=encoder_dim)
# 输入: (B, 1, T, input_dim), 输出: (B, T/4, output_dim)

# 掩码卷积（处理变长序列）
MaskConv2d(nn.Sequential(...))
```

### 掩码工具

```python
# 填充掩码（encoder 侧）
get_attn_pad_mask(inputs: Tensor, input_lengths: Tensor, expand_length: int) -> Tensor

# 因果掩码（decoder 侧）
get_attn_subsequent_mask(seq: Tensor) -> Tensor
```

---

## 8. 损失函数（Criterion）

所有损失类继承自 `nn.Module`，接受 `(configs, tokenizer)` 构造：

### CTCLoss

```python
CTCLoss(configs, tokenizer)
forward(encoder_log_probs, targets, encoder_output_lengths, target_lengths) -> Tensor
```

- 封装 `nn.CTCLoss`
- 参数: `reduction='mean'`, `zero_infinity=True`

### CrossEntropyLoss

```python
CrossEntropyLoss(configs, tokenizer)
forward(logits, targets, input_lengths, target_lengths) -> Tensor
```

- 封装 `nn.CrossEntropyLoss`
- 自动忽略 `pad_id`

### 联合损失（在模型内实现）

```python
# ASREncoderDecoderModel.collect_outputs() 中
loss = ctc_weight * ctc_loss + (1 - ctc_weight) * ce_loss
```

---

## 9. 解码策略（Search）

| 类 | 适用模型 | 算法 |
|----|----------|------|
| `BeamSearchLSTM` | LAS, Conformer+LSTM | 标准 beam search |
| `BeamSearchCTC` | CTC 模型 | prefix beam search（依赖 `ctcdecode`），支持 LM 融合 |
| `BeamSearchRNNTransducer` | RNN Transducer | 时间同步 beam search |
| `BeamSearchTransformerTransducer` | Transformer Transducer | 同上 |

**Transducer Beam Search 关键参数**:
```python
BeamSearchRNNTransducer(
    decoder,
    joint,
    blank_id,
    beam_size=10,
    expand_beam=2.3,   # 扩展候选的分数阈值
    state_beam=4.6,    # 状态剪枝阈值
    vocab_size=...
)
```

---

## 10. 训练框架

### ASRModel（Lightning 基类）

```python
class ASRModel(pl.LightningModule):
    def configure_optimizers(self):
        # 根据 configs.model.optimizer 实例化优化器
        # 根据 configs.lr_scheduler 实例化调度器
        # 返回 {"optimizer": ..., "lr_scheduler": ...}

    def configure_criterion(self) -> nn.Module:
        # 根据 configs.criterion 类型实例化损失函数

    def training_step(self, batch, batch_idx): ...
    def validation_step(self, batch, batch_idx): ...
    def test_step(self, batch, batch_idx): ...
```

### ASREncoderDecoderModel

```python
def collect_outputs(self, phase, loss, logits, targets, input_lengths,
                    target_lengths, encoder_logits=None) -> OrderedDict:
    # 计算 WER/CER，记录到 Lightning 日志
    # 返回包含 loss, wer, cer 的字典
```

### 训练流程（以 Conformer 为例）

```
batch = (audio_features, audio_lengths, labels, label_lengths)
  ↓
ConformerEncoder(audio_features, audio_lengths)
  → encoder_outputs (B, T', 512), encoder_logits (optional), output_lengths
  ↓
LSTMAttentionDecoder(encoder_outputs, output_lengths, labels, ...)
  → decoder_outputs (B, T, num_classes)
  ↓
CrossEntropyLoss(decoder_outputs, labels, ...)  [+ CTCLoss if joint]
  ↓
loss.backward() → optimizer.step() → scheduler.step()
  ↓
validation: teacher_forcing_ratio=0.0, beam search decoding
  → WER / CER logging
```

### Callbacks

```python
# 每 N 步保存 checkpoint
CheckpointEveryNSteps(save_step_frequency=configs.trainer.save_checkpoint_n_steps)
```

---

## 11. 数据与特征提取

### 音频特征

| 特征 | 参数 | 输出维度 |
|------|------|---------|
| MelSpectrogram | `sample_rate=16000`, `n_mels=80` | 80 |
| FilterBank | `sample_rate=16000`, `n_mels=80` | 80 |
| Spectrogram | `sample_rate=16000`, `n_fft=512` | n_fft/2 + 1 |
| MFCC | `sample_rate=16000`, `n_mfcc=40` | 40 |

### 数据增强（SpecAugment 等）

```python
AugmentConfigs:
    freq_mask_para: int = 27        # 频率掩码宽度上限
    freq_mask_num: int = 2          # 频率掩码数量
    time_mask_num: int = 10         # 时间掩码数量
    noise_level: float = 0.0        # 噪声注入强度
```

### DataModule

```python
LightningLibriSpeechDataModule(configs, tokenizer)
# 自动下载、解压、生成 manifest CSV
# train_dataloader() / val_dataloader() / test_dataloader()
# 支持 SmartBatchingSampler（按长度排序，减少 padding）
```

---

## 12. Tokenizer

**抽象接口**：

```python
class Tokenizer(ABC):
    def encode(self, labels: str) -> List[int]: ...
    def decode(self, labels: Tensor) -> str: ...
    def load_vocab(self) -> dict: ...

    @property
    def sos_id(self) -> int: ...   # Start-of-sequence token ID
    @property
    def eos_id(self) -> int: ...   # End-of-sequence token ID
    @property
    def pad_id(self) -> int: ...   # Padding token ID
    @property
    def blank_id(self) -> int: ... # CTC blank token ID
    def __len__(self) -> int: ...  # Vocabulary size
```

**实现**：
- `LibriSpeechCharacterTokenizer`: 从 CSV 文件加载字符词表
- `LibriSpeechSubwordTokenizer`: SentencePiece / BPE 子词分词

---

## 13. 语言模型（LM）

继承自 `ASRLanguageModelBase(pl.LightningModule)`：

### LSTMForLanguageModel

```
Token → Embedding → LSTM (num_layers) → Linear → logits
```

### TransformerForLanguageModel

```
Token → TransformerEmbedding → TransformerForLanguageModelLayer × n → Linear → logits
```

**LM 用途**：
- 独立训练（用于语言建模任务）
- 与 `BeamSearchCTC` 融合（shallow fusion），参数 `alpha`（LM 权重）、`beta`（词插入惩罚）

---

## 14. 优化器与调度器

### 支持的优化器

在 `configs.model.optimizer` 中指定（字符串）：

| 名称 | 来源 |
|------|------|
| `Adam`, `AdamW`, `SGD`, `ASGD`, `Adagrad`, `Adadelta`, `Adamax` | PyTorch 内置 |
| `RAdam` | `asr/optim/radam.py` |
| `AdamP` | `asr/optim/adamp.py` |
| `Novograd` | `asr/optim/novograd.py` |

### TransformerLRScheduler（推荐）

三阶段调度：

```
阶段 0: 线性预热 0 → peak_lr（warmup_steps 步）
阶段 1: 指数衰减 peak_lr → final_lr（decay_steps 步）
         lr = peak_lr * exp(-decay_factor * step)
阶段 2: 保持 final_lr
```

**配置示例**:
```yaml
lr_scheduler:
  scheduler_type: transformer
  peak_lr: 0.05
  final_lr: 0.0001
  warmup_steps: 10000
  decay_steps: 200000
```

---

## 15. 评估指标

```python
from asr.metrics.metrics import CharacterErrorRate, WordErrorRate

cer = CharacterErrorRate(tokenizer)
wer = WordErrorRate(tokenizer)

# 累积计算
cer.update(targets, predictions)
cer.compute()  # 返回当前累积 CER

# 重置
cer.reset()
```

两者都基于 **Levenshtein 编辑距离**，`CER` 在字符级计算，`WER` 在空格分隔的词级计算。

---

## 16. 依赖库

**核心依赖** (requirements.txt):

```
torch==1.13.1+cu117
pytorch-lightning==1.6.0
torchaudio==0.13.1+cu117
librosa==0.9.2
hydra-core==1.0.7
omegaconf==2.0.6
tokenizers==0.12.1
sentencepiece==0.1.99
Levenshtein==0.20.5
torchmetrics==0.6.0
wandb==0.15.12
pandas==1.5.0
```

**可选依赖**（未在 requirements.txt 中，但代码中引用）：
- `ctcdecode`：CTC beam search with LM（`BeamSearchCTC` 使用）
- `warp-rnnt`：高效 RNN-T Loss（`ASRTransducerModel` 使用）

---

## 17. 新增模型指南

以下是在本框架中新增一个模型（如 Whisper 风格的 encoder-only CTC 模型）的完整步骤：

### Step 1: 定义配置类

```python
# asr/dataclass/configurations.py
@dataclass
class MyNewModelConfigs(ASRDataclass):
    model_name: str = field(default="my_new_model", metadata={"help": "Model name"})
    encoder_dim: int = field(default=512, metadata={"help": "Encoder dimension"})
    num_layers: int = field(default=12, metadata={"help": "Number of encoder layers"})
    # ... 其他参数

MODEL_DATACLASS_REGISTRY["my_new_model"] = MyNewModelConfigs
```

### Step 2: 注册 Hydra 配置

```python
# asr/dataclass/initialize.py → hydra_train_init()
cs.store(group="model", name="my_new_model", node=MyNewModelConfigs)
```

### Step 3: 实现编码器（如有需要）

```python
# asr/encoders/my_encoder.py
from asr.encoders.asr_encoder import ASREncoder

class MyEncoder(ASREncoder):
    def __init__(self, configs) -> None:
        super().__init__()
        # ...

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # 返回 (encoder_outputs, ctc_logits_or_None, output_lengths)
        ...
```

### Step 4: 实现模型

根据解码方式选择继承的基类：

```python
# asr/models/my_model/model.py
from asr.models.asr_encoder_decoder import ASREncoderDecoderModel  # 有 decoder
# 或
from asr.models.asr_ctc_model import ASRCTCModel                   # CTC only
# 或
from asr.models.asr_transducer_model import ASRTransducerModel     # Transducer

class MyNewModel(ASREncoderDecoderModel):
    def __init__(self, configs, tokenizer) -> None:
        super().__init__(configs, tokenizer)
        self.encoder = MyEncoder(configs)
        self.decoder = LSTMAttentionDecoder(...)  # 或自定义 decoder

    def forward(self, inputs: Tensor, input_lengths: Tensor,
                targets: Optional[Tensor] = None,
                teacher_forcing_ratio: float = 1.0) -> Tuple[Tensor, ...]:
        encoder_outputs, encoder_logits, output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(encoder_outputs, output_lengths, targets, ...)
        return encoder_outputs, encoder_logits, decoder_outputs

    def training_step(self, batch, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch
        encoder_outputs, encoder_logits, decoder_outputs = self(
            inputs, input_lengths, targets, teacher_forcing_ratio=1.0
        )
        return self.collect_outputs("train", loss, decoder_outputs, targets, ...)

    def validation_step(self, batch, batch_idx: int):
        # teacher_forcing_ratio=0.0 for greedy/beam decoding
        ...

    def test_step(self, batch, batch_idx: int):
        ...

    def set_beam_decoder(self, beam_size: int = 3) -> None:
        self.decode = BeamSearchLSTM(self.decoder, beam_size, ...)
```

### Step 5: 注册模型到 `__init__.py`

```python
# asr/models/__init__.py 或对应位置
from asr.models.my_model.model import MyNewModel
```

### Step 6: 创建 YAML 配置

```yaml
# asr/configs/model/my_new_model.yaml
model_name: my_new_model
encoder_dim: 512
num_layers: 12
```

### Step 7: 训练

```bash
python train.py model=my_new_model trainer=gpu batch_size=32
```

---

## 18. MLC 集成指南

[MLC-LLM (Machine Learning Compilation)](https://github.com/mlc-ai/mlc-llm) 允许将 PyTorch 模型编译为跨平台高性能推理引擎（CUDA、Metal、WebGPU、Android/iOS 等）。

### 18.1 MLC 工作原理

```
PyTorch Model
    ↓ torch.export / torch.fx
TVM Relax IR (中间表示)
    ↓ TVM 编译优化
    │   - Operator fusion
    │   - Memory planning
    │   - Quantization (INT4/INT8/FP16)
    ↓
Native binary (.so / .wasm / .apk)
    ↓
mlc_chat / mlc_runtime 推理
```

### 18.2 集成准备

**模型要求**：
1. 推理路径必须是**静态图**（无 Python 控制流依赖 tensor 值）
2. 所有形状最好在编译时确定，或使用**动态形状**注解
3. 去掉训练相关代码（dropout、teacher forcing 等）

**为 MLC 创建推理专用 module**：

```python
# asr/models/conformer/mlc_model.py
import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class ConformerEncoderInference(nn.Module):
    """推理专用 Conformer 编码器（无 dropout，确定性输出）"""

    def __init__(self, conformer_encoder):
        super().__init__()
        self.encoder = conformer_encoder
        self.encoder.eval()

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            outputs, _, output_lengths = self.encoder(inputs, input_lengths)
        return outputs, output_lengths


class ConformerCTCInference(nn.Module):
    """CTC 推理（Conformer + CTC head，无 decoder）"""

    def __init__(self, encoder, ctc_head):
        super().__init__()
        self.encoder = encoder
        self.ctc_head = ctc_head

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tensor:
        outputs, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.ctc_head(outputs)   # (B, T', num_classes)
        return logits.log_softmax(dim=-1)
```

### 18.3 导出为 TorchScript（MLC 前置步骤）

```python
# scripts/export_mlc.py
import torch
from asr.models.conformer.models import ConformerLSTMModel
from asr.models.conformer.mlc_model import ConformerCTCInference

# 1. 加载训练好的模型
model = ConformerLSTMModel.load_from_checkpoint("checkpoint.ckpt", ...)
model.eval()

# 2. 创建推理包装
encoder_inference = ConformerCTCInference(model.encoder, model.encoder.fc)

# 3. 用 torch.export 导出（MLC 推荐路径）
example_inputs = (
    torch.randn(1, 300, 80),   # (batch, time, mels)
    torch.tensor([300]),        # input_lengths
)

# 方式 A: torch.export（推荐，支持动态形状）
exported = torch.export.export(
    encoder_inference,
    args=example_inputs,
    dynamic_shapes={
        "inputs": {0: torch.export.Dim("batch"), 1: torch.export.Dim("time")},
        "input_lengths": {0: torch.export.Dim("batch")},
    }
)
torch.export.save(exported, "conformer_encoder.pt2")

# 方式 B: TorchScript（兼容旧版 MLC）
scripted = torch.jit.trace(encoder_inference, example_inputs)
scripted.save("conformer_encoder_traced.pt")
```

### 18.4 MLC 编译步骤

```bash
# 安装 MLC
pip install mlc-ai-nightly  # 或从源码编译

# 方式 A: 使用 mlc_chat 转换（适合 LLM 风格模型）
# 方式 B: 使用 TVM Python API（更通用）

python -c "
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program
import torch

# 加载 torch.export 产物
exported = torch.export.load('conformer_encoder.pt2')

# 转换到 TVM Relax IR
mod = from_exported_program(exported, keep_params_as_input=True)

# 优化 + 编译（以 CUDA 为例）
target = tvm.target.Target('cuda')
with tvm.transform.PassContext(opt_level=3):
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)

# 构建
ex = relax.build(mod, target)
ex.export_library('conformer_encoder_cuda.so')
print('Compiled successfully!')
"
```

### 18.5 MLC 运行时推理

```python
import tvm
from tvm import relax
import numpy as np

# 加载编译后的库
lib = tvm.runtime.load_module("conformer_encoder_cuda.so")
dev = tvm.cuda(0)

# 创建 VM
vm = relax.VirtualMachine(lib, dev)

# 推理
inputs = tvm.nd.array(np.random.randn(1, 300, 80).astype("float32"), dev)
input_lengths = tvm.nd.array(np.array([300], dtype="int64"), dev)

outputs = vm["forward"](inputs, input_lengths)
encoder_out = outputs[0].numpy()  # (1, 75, 512)
```

### 18.6 量化（INT8/INT4）

```python
# MLC 量化（编译时）
from tvm.relax.quantize import quantize

# INT8 量化
mod_quant = quantize(mod, quantization_scheme="int8")

# INT4 量化（更激进，适合移动端）
mod_quant = quantize(mod, quantization_scheme="int4")
```

### 18.7 集成建议与注意事项

**推荐的 MLC 集成方案**：

| 组件 | 建议 | 原因 |
|------|------|------|
| Conformer 编码器 | 完整编译 | 纯前馈，无控制流，MLC 友好 |
| CTC 解码（greedy） | 完整编译 | `argmax` 可编译 |
| CTC 束搜索 | Python/C++ 侧实现 | 动态控制流，编译困难 |
| LSTM 注意力解码器 | 循环展开或单步编译 | 自回归逐步解码，需要单步推理接口 |
| Transformer 解码器 | KV Cache + 单步编译 | 类似 LLM 推理模式 |
| Transducer Joint | 完整编译 | 矩阵运算，MLC 友好 |

**控制流问题处理**：
```python
# 对于 LSTM 解码器，提供单步推理接口
class LSTMDecoderStep(nn.Module):
    """单步 LSTM 解码（用于 MLC 逐步推理）"""
    def forward(
        self,
        input_token: Tensor,        # (B, 1) 当前 token
        hidden: Tensor,             # (num_layers, B, hidden_dim) LSTM hidden state
        cell: Tensor,               # (num_layers, B, hidden_dim) LSTM cell state
        encoder_outputs: Tensor,    # (B, T', encoder_dim) 编码器输出（缓存）
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # 返回 (logits, new_hidden, new_cell)
        ...
```

**动态长度处理**：
- MLC 推荐使用 **padding + masking** 而非动态长度
- 导出时使用 `torch.export.Dim` 声明动态维度
- 推理时固定 batch_size=1（适合流式/实时场景）

### 18.6 推荐集成路径

```
训练: 当前 PyTorch + Lightning 框架（不变）
        ↓ checkpoint
推理导出:
  1. 提取编码器 + CTC head（或 Joint Network）
  2. 包装为推理专用 nn.Module（去掉 dropout 等）
  3. torch.export 导出 .pt2 文件
        ↓
MLC 编译:
  4. TVM from_exported_program 加载
  5. 优化 pass（算子融合、内存规划）
  6. 量化（可选，INT8/INT4）
  7. 编译为目标平台 .so
        ↓
部署:
  8. 服务端: TVM runtime Python/C++ API
  9. 移动端: mlc-chat Android/iOS runtime
  10. 浏览器: WebGPU backend
```

---

*文档生成时间: 2026-03-20 | 分支: claude/bug-fixes-types-docstrings*
