import numpy as np
from omegaconf import DictConfig
from torch import Tensor

from ....utils import TORCHAUDIO_IMPORT_ERROR
from ... import register_audio_feature_transform
from ...audio.filter_bank.configuration import FilterBankConfigs


@register_audio_feature_transform("fbank", dataclass=FilterBankConfigs)
class FilterBankFeatureTransform(object):
    def __init__(self, configs: DictConfig) -> None:
        super(FilterBankFeatureTransform, self).__init__()
        try:
            import torchaudio
        except ImportError:
            raise ImportError(TORCHAUDIO_IMPORT_ERROR)

        self.num_mels = configs.audio.num_mels
        self.frame_length = configs.audio.frame_length
        self.frame_shift = configs.audio.frame_shift
        self.function = torchaudio.compliance.kaldi.fbank

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return (
            self.function(
                Tensor(signal).unsqueeze(0),
                num_mel_bins=self.num_mels,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
            )
            .transpose(0, 1)
            .numpy()
        )