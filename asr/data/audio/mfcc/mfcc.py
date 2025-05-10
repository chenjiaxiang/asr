import numpy as np
from omegaconf import DictConfig

from ....utils import LIBROSA_IMPORT_ERROR
from ... import register_audio_feature_transform
from ...audio.mfcc.configuration import MFCCConfigs

@register_audio_feature_transform("mfcc", dataclass=MFCCConfigs)
class MFCCFeatureTransform(object):
    def __init__(self, configs: DictConfig) -> None:
        super(MFCCFeatureTransform, self).__init__()
        try:
            import librosa
        except ImportError:
            raise ImportError(LIBROSA_IMPORT_ERROR)

        self.sample_rate = configs.audio.sample_rate
        self.num_mels = configs.audio.num_mels
        self.n_fft = int(round(configs.audio.sample_rate * 0.001 * configs.audio.frame_length))
        self.hop_length = int(round(configs.audio.sample_rate * 0.001 * configs.audio.frame_shift))
        self.function = librosa.feature.mfcc

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        mfcc = self.function(
            y=signal,
            sr=self.sample_rate,
            n_mfcc=self.num_mels,
            # n_fft=self.n_fft,    # TODO, the installed librosa.feature.mfcc missing n_fft parameter. 
            hop_length=self.hop_length,
        )
        return mfcc