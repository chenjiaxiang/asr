import numpy as np
from omegaconf import DictConfig
from torch import Tensor

from ....utils import LIBROSA_IMPORT_ERROR
from ... import register_audio_feature_transform
from ...audio.melspectrogram.configuration import MelSpectrogramConfigs

@register_audio_feature_transform("melspectrogram", dataclass=MelSpectrogramConfigs)
class MelSpectrogramFeatureTransform(object):
    def __init__(self, configs: DictConfig) -> None:
        super(MelSpectrogramFeatureTransform, self).__init__()
        try:
            import librosa
        except ImportError:
            raise ImportError(LIBROSA_IMPORT_ERROR)

        self.sample_rate = configs.audio.sample_rate
        self.num_mels = configs.audio.num_mels
        self.n_fft = int(round(configs.audio.sample_rate * 0.001 * configs.audio.frame_length))
        self.hop_length = int(round(configs.audio.sample_rate * 0.001 * configs.audio.frame_shift))
        self.function = librosa.feature.melspectrogram
        self.power_to_db = librosa.power_to_db

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        melspectrogram = self.function(
            y=signal,
            sr=self.sample_rate,
            # n_mels=self.num_mels,  # TODO, the installed librosa.feature.melspectrogram missing n_mels parameter.
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        melspectrogram = self.power_to_db(melspectrogram, ref=np.max)
        return melspectrogram