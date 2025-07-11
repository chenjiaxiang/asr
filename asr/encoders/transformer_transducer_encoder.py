from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from asr.encoders import ASREncoder
from asr.modules import MultiHeadAttention, PositionalEncoding, PositionwiseFeedForward, get_attn_pad_mask