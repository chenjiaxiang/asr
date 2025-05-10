from typing import List, Tuple
import logging
import os
import random

import librosa
import numpy as np
from torch import Tensor

from ..audio.load import load_audio

logger = logging.getLogger(__name__)


class SpecAugment(object):
    def __init__(self, freq_mask_para: int = 18, time_mask_num: int = 10, freq_mask_num: int = 2) -> None:
        self.freq_mask_para = freq_mask_para
        self.time_mask_num = time_mask_num
        self.freq_mask_num = freq_mask_num

    def __call__(self, feature: Tensor) -> Tensor:
        """Provides SpecAugmentation for audio"""
        time_axis_length = feature.size(0)
        freq_axis_length = feature.size(1)
        time_mask_para = time_axis_length / 20 # Refer to "Specaugment on large scale dataset" paper

        # time mask
        for _ in range(self.time_mask_num):
            t = int(np.random.uniform(low=0.0, high=time_mask_para))
            t0 = random.randint(0, time_axis_length - t)
            feature[t0 : t0 + t, :] = 0

        # freq mask
        for _ in range(self.freq_mask_num):
            f = int(np.random.uniform(low=0.0, high=self.freq_mask_para))
            f0 = random.randint(0, freq_axis_length - f)
            feature[:, f0 : f0 + f] = 0
        
        return feature


class NoiseInjector(object):
    def __init__(
            self,
            noise_dataset_dir: str,
            sample_rate: int = 16000,
            noise_level: float = 0.7,
    ) -> None:
        if not os.path.exists(noise_dataset_dir):
            logger.info("Directory doesn't exist: {0}".format(noise_dataset_dir))
            raise IOError
        
        logger.info("Create Noise Injector...")

        self.sample_rate = sample_rate
        self.noise_level = noise_level
        self._load_audio = load_audio
        self.audio_paths = self.create_audio_paths(noise_dataset_dir)
        self.dataset = self.create_noiseset(noise_dataset_dir)

        logger.info("Create Noise injector compelete !!")

    def __call__(self, signal):
        noise = np.random.choice(self.dataset)
        noise_level = np.random.uniform(0, self.noise_level)

        signal_length = len(signal)
        noise_length = len(noise)

        if signal_length >= noise_length:
            noise_start = int(np.random.rand() * (signal_length - noise_length))
            noise_end = int(noise_start + noise_length)
            signal[noise_start: noise_end] += noise * noise_level
        else:
            signal += noise[:signal_length] * noise_level

    def create_audio_paths(self, dataset_path: str) -> List[str]:
        audio_paths = list()
        noise_audio_paths = os.listdir(dataset_path)
        num_noise_audio_data = len(noise_audio_paths)

        for idx in range(num_noise_audio_data):
            if (
                noise_audio_paths[idx].endswith(".pcm")
                or noise_audio_paths[idx].endswith(".wav")
                or noise_audio_paths[idx].endswith(".flac")
            ):
                audio_paths.append(noise_audio_paths[idx])
        
        return audio_paths

    def create_noiseset(self, dataset_path: str) -> List[np.ndarray]:
        dataset = list()

        for audio_path in self.audio_paths:
            audio_path = os.path.join(dataset_path, audio_path)
            noise = self._load_audio(audio_path, self.sample_rate, del_silence=False)    

            if noise is not None:
                dataset.append(noise)

        return dataset


class TimeStretchAugment(object):
    def __init__(self, min_rate: float = 0.7, max_rate: float = 1.4) -> None:
        super(TimeStretchAugment, self).__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return librosa.effects.time_stretch(signal, random.uniform(self.min_rate, self.max_rate))

    
class JoiningAugment(object):
    def __init__(self):
        super(JoiningAugment, self).__init__()
    
    def __call__(self, signals: Tuple[np.ndarray]) -> np.ndarray:
        return np.concatenate([signal for signal in signals])