# ASR — Automatic Speech Recognition Framework

> **Language / 语言 / 言語**: [English](#english) | [中文](#中文) | [日本語](#日本語)

---

## English

A modular, research-friendly **Automatic Speech Recognition (ASR)** framework built on **PyTorch** and **PyTorch Lightning**, with flexible configuration via **Hydra + OmegaConf**.

### Supported Architectures

| Model | Encoder | Decoder | Loss |
|-------|---------|---------|------|
| Conformer + LSTM | ConformerEncoder | LSTMAttentionDecoder | CE / CTC+CE |
| Transformer | TransformerEncoder | TransformerDecoder | CE / CTC+CE |
| LAS (Listen, Attend, Spell) | LSTMEncoder | LSTMAttentionDecoder | CE / CTC+CE |
| LAS + Location-Aware Attention | LSTMEncoder | LocationAwareAttention | CE / CTC+CE |
| Transformer Transducer | TransformerTransducerEncoder | TransformerTransducerDecoder | RNN-T |
| CTC | ConformerEncoder / LSTMEncoder | — | CTC |
| Transformer LM | — | TransformerDecoder | CE (LM only) |

### Features

- **Modular design** — encoders, decoders, and attention modules are independently composable
- **Multiple attention mechanisms** — Dot-product, Additive (Bahdanau), Multi-head, Relative multi-head, Location-aware
- **Conformer blocks** — Conv + Self-attention with macaron-style feed-forward (Gulati et al., 2020)
- **Beam search decoding** — CTC, LSTM, RNN-Transducer, Transformer-Transducer variants
- **Joint CTC-Attention training** — optional auxiliary CTC loss on the encoder
- **Hydra config system** — swap models, datasets, optimizers, and tokenizers via YAML or CLI flags
- **PyTorch Lightning** — built-in multi-GPU, mixed-precision, and checkpointing
- **Language model support** — LSTM-LM and Transformer-LM for shallow fusion

### Installation

```bash
git clone https://github.com/chenjiaxiang/asr.git
cd asr
pip install -r requirements.txt
```

### Quick Start

**Training (LibriSpeech, Conformer+LSTM):**
```bash
bash train_librispeech.sh
# or with Hydra overrides:
python -m asr_cli.train model=conformer_lstm dataset=librispeech trainer=gpu
```

**Configuration override example:**
```bash
python -m asr_cli.train \
  model=conformer_lstm \
  model.encoder_dim=256 \
  model.num_encoder_layers=12 \
  criterion=joint_ctc_cross_entropy \
  criterion.ctc_weight=0.3
```

### Project Structure

```
asr/
├── configs/          # Default YAML configs (train.yaml, eval.yaml)
├── dataclass/        # Hydra-registered dataclasses (configurations.py)
├── criterion/        # CTCLoss, CrossEntropyLoss
├── data/             # Audio feature extraction (MelSpec, MFCC, FilterBank)
├── datasets/         # LightningDataModules (LibriSpeech, AISHELL)
├── decoders/         # LSTMAttentionDecoder, TransformerDecoder, RNN-T
├── encoders/         # ConformerEncoder, LSTMEncoder, Transformer-T
├── lm/               # LSTM-LM, Transformer-LM
├── metrics/          # WER, CER (Levenshtein-based)
├── models/           # Full LightningModules (conformer/, transformer/, ...)
├── modules/          # Reusable building blocks (attention, conv, PE, ...)
├── optim/            # RAdam, AdamP, NovoGrad + LR schedulers
├── search/           # Beam search variants
└── tokenizers/       # Character-level, Subword (BPE/SentencePiece)
```

For a full architecture overview, developer guide, and MLC integration notes, see [PROJECT.md](PROJECT.md).

---

## 中文

基于 **PyTorch** 和 **PyTorch Lightning** 构建的模块化 ASR（自动语音识别）研究框架，通过 **Hydra + OmegaConf** 实现灵活配置。

### 支持的模型架构

| 模型 | 编码器 | 解码器 | 损失函数 |
|------|--------|--------|---------|
| Conformer + LSTM | ConformerEncoder | LSTMAttentionDecoder | CE / CTC+CE |
| Transformer | TransformerEncoder | TransformerDecoder | CE / CTC+CE |
| LAS（Listen, Attend, Spell） | LSTMEncoder | LSTMAttentionDecoder | CE / CTC+CE |
| LAS + 位置感知注意力 | LSTMEncoder | LocationAwareAttention | CE / CTC+CE |
| Transformer Transducer | TransformerTransducerEncoder | TransformerTransducerDecoder | RNN-T |
| CTC | ConformerEncoder / LSTMEncoder | — | CTC |
| Transformer 语言模型 | — | TransformerDecoder | CE（仅语言模型） |

### 主要特性

- **模块化设计** — 编码器、解码器、注意力模块可自由组合
- **多种注意力机制** — 点积注意力、加性注意力（Bahdanau）、多头注意力、相对位置多头注意力、位置感知注意力
- **Conformer 块** — 卷积 + 自注意力，采用 Macaron 风格前馈网络（Gulati et al., 2020）
- **Beam Search 解码** — 支持 CTC、LSTM、RNN-Transducer、Transformer-Transducer 多种变体
- **联合 CTC-注意力训练** — 可在编码器上附加辅助 CTC 损失
- **Hydra 配置系统** — 通过 YAML 文件或命令行参数灵活切换模型、数据集、优化器和分词器
- **PyTorch Lightning** — 原生支持多 GPU、混合精度训练和断点续训
- **语言模型支持** — LSTM-LM 和 Transformer-LM，可用于浅层融合解码

### 安装

```bash
git clone https://github.com/chenjiaxiang/asr.git
cd asr
pip install -r requirements.txt
```

### 快速开始

**训练（LibriSpeech，Conformer+LSTM）：**
```bash
bash train_librispeech.sh
# 或使用 Hydra 命令行覆盖参数：
python -m asr_cli.train model=conformer_lstm dataset=librispeech trainer=gpu
```

**配置覆盖示例：**
```bash
python -m asr_cli.train \
  model=conformer_lstm \
  model.encoder_dim=256 \
  model.num_encoder_layers=12 \
  criterion=joint_ctc_cross_entropy \
  criterion.ctc_weight=0.3
```

### 项目结构

```
asr/
├── configs/          # 默认 YAML 配置（train.yaml, eval.yaml）
├── dataclass/        # Hydra 注册的数据类（configurations.py）
├── criterion/        # CTCLoss, CrossEntropyLoss
├── data/             # 音频特征提取（梅尔频谱、MFCC、滤波器组）
├── datasets/         # LightningDataModule（LibriSpeech、AISHELL）
├── decoders/         # LSTMAttentionDecoder, TransformerDecoder, RNN-T
├── encoders/         # ConformerEncoder, LSTMEncoder, Transformer-T
├── lm/               # LSTM 语言模型, Transformer 语言模型
├── metrics/          # WER、CER（基于 Levenshtein 距离）
├── models/           # 完整 LightningModule（conformer/、transformer/ 等）
├── modules/          # 可复用基础组件（注意力、卷积、位置编码等）
├── optim/            # RAdam、AdamP、NovoGrad 及学习率调度器
├── search/           # 多种 Beam Search 实现
└── tokenizers/       # 字符级分词器、子词分词器（BPE/SentencePiece）
```

完整的架构说明、开发指南及 MLC 集成方案，请参阅 [PROJECT.md](PROJECT.md)。

---

## 日本語

**PyTorch** と **PyTorch Lightning** をベースにした、**Hydra + OmegaConf** による柔軟な設定システムを備えたモジュール型 ASR（自動音声認識）研究フレームワークです。

### 対応アーキテクチャ

| モデル | エンコーダ | デコーダ | 損失関数 |
|--------|-----------|---------|---------|
| Conformer + LSTM | ConformerEncoder | LSTMAttentionDecoder | CE / CTC+CE |
| Transformer | TransformerEncoder | TransformerDecoder | CE / CTC+CE |
| LAS（Listen, Attend, Spell） | LSTMEncoder | LSTMAttentionDecoder | CE / CTC+CE |
| LAS + 位置認識アテンション | LSTMEncoder | LocationAwareAttention | CE / CTC+CE |
| Transformer Transducer | TransformerTransducerEncoder | TransformerTransducerDecoder | RNN-T |
| CTC | ConformerEncoder / LSTMEncoder | — | CTC |
| Transformer 言語モデル | — | TransformerDecoder | CE（言語モデルのみ） |

### 主な特徴

- **モジュール設計** — エンコーダ・デコーダ・アテンションモジュールを自由に組み合わせ可能
- **多様なアテンション機構** — ドット積、加法的（Bahdanau）、マルチヘッド、相対位置マルチヘッド、位置認識アテンション
- **Conformer ブロック** — マカロン型フィードフォワードを持つ Conv + Self-Attention（Gulati et al., 2020）
- **ビームサーチデコーディング** — CTC・LSTM・RNN-Transducer・Transformer-Transducer に対応
- **CTC-Attention 結合学習** — エンコーダへの補助 CTC 損失を任意で追加可能
- **Hydra 設定システム** — YAML またはコマンドライン引数でモデル・データセット・最適化手法・トークナイザを柔軟に切り替え
- **PyTorch Lightning** — マルチ GPU・混合精度・チェックポイント再開に標準対応
- **言語モデルサポート** — LSTM-LM と Transformer-LM によるシャロー・フュージョンデコード

### インストール

```bash
git clone https://github.com/chenjiaxiang/asr.git
cd asr
pip install -r requirements.txt
```

### クイックスタート

**学習（LibriSpeech、Conformer+LSTM）：**
```bash
bash train_librispeech.sh
# または Hydra でパラメータをオーバーライド：
python -m asr_cli.train model=conformer_lstm dataset=librispeech trainer=gpu
```

**設定オーバーライド例：**
```bash
python -m asr_cli.train \
  model=conformer_lstm \
  model.encoder_dim=256 \
  model.num_encoder_layers=12 \
  criterion=joint_ctc_cross_entropy \
  criterion.ctc_weight=0.3
```

### プロジェクト構成

```
asr/
├── configs/          # デフォルト YAML 設定（train.yaml, eval.yaml）
├── dataclass/        # Hydra 登録済みデータクラス（configurations.py）
├── criterion/        # CTCLoss, CrossEntropyLoss
├── data/             # 音声特徴抽出（メルスペクトログラム、MFCC、フィルタバンク）
├── datasets/         # LightningDataModule（LibriSpeech、AISHELL）
├── decoders/         # LSTMAttentionDecoder, TransformerDecoder, RNN-T
├── encoders/         # ConformerEncoder, LSTMEncoder, Transformer-T
├── lm/               # LSTM 言語モデル、Transformer 言語モデル
├── metrics/          # WER・CER（Levenshtein 距離ベース）
├── models/           # 完全な LightningModule（conformer/、transformer/ など）
├── modules/          # 再利用可能なビルディングブロック（アテンション、畳み込み、PE など）
├── optim/            # RAdam、AdamP、NovoGrad および学習率スケジューラ
├── search/           # 各種ビームサーチ実装
└── tokenizers/       # 文字レベル・サブワード（BPE/SentencePiece）トークナイザ
```

アーキテクチャの詳細・開発ガイド・MLC 統合については [PROJECT.md](PROJECT.md) を参照してください。
