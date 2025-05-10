import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

from ....utils import TORCHAUDIO_IMPORT_ERROR
from ... import register_audio_feature_transform
from ...audio.spectrogram.configuration import SpectorgramConfigs


@register_audio_feature_transform("spectrogram", dataclass=SpectorgramConfigs)
class SpectrogramFeatureTransform(object):
    def __init__(self, configs: DictConfig) -> None:
        super(SpectrogramFeatureTransform, self).__init__()
        try:
            import torchaudio
        except ImportError:
            raise ImportError(TORCHAUDIO_IMPORT_ERROR)

        self.n_fft = int(round(configs.audio.sample_rate * 0.001 * configs.audio.frame_length))
        self.hop_length = int(round(configs.audio.sample_rate * 0.001 * configs.audio.frame_shift))
        self.function = torch.stft

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        spectrogram = self.function(
            input=Tensor(signal),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hamming_window(self.n_fft),
            center=False,
            normalized=False,
            onesided=True,
        )
        spectrogram = (spectrogram[:, :, 0].pow(2) + spectrogram[:, :, 1].pow(2)).pow(0.5)
        spectrogram = np.log1p(spectrogram.numpy())
        return spectrogram