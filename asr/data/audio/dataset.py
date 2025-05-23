import logging
import os
import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from asr.data import AUDIO_FEATURE_TRANSFORM_REGISTRY
from asr.data.audio.augment import JoiningAugment, NoiseInjector, SpecAugment, TimeStretchAugment
from asr.data.audio.load import load_audio

logger = logging.getLogger(__name__)

class SpeechToTextDataset(Dataset):
    NONE_AUGMENT = 0
    SPEC_AUGMENT = 1
    NOISE_AUGMENT = 2
    TIME_AUGMENT = 3
    AUDIO_JOINING = 4

    def __init__(
            self,
            configs: DictConfig,
            dataset_path: str,
            audio_paths: list,
            transcripts: list,
            sos_id: int = 1,
            eos_id: int = 2,
            del_silence: bool = False,
            apply_spec_augment: bool = False,
            apply_noise_augment: bool = False,
            apply_time_stretch_augment: bool = False,
            apply_joining_augment: bool = False,
    ) -> None:
        super(SpeechToTextDataset, self).__init__()
        self.dataset_path = dataset_path
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.augments = [self.NONE_AUGMENT] * len(self.audio_paths)
        self.dataset_size = len(self.audio_paths)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.sample_rate = configs.audio.sample_rate
        self.num_mels = configs.audio.num_mels
        self.del_silence = del_silence
        self.apply_spec_augment = apply_spec_augment
        self.apply_noise_augment = apply_noise_augment
        self.apply_time_stretch_augment = apply_time_stretch_augment
        self.apply_joining_augment = apply_joining_augment
        self.transforms = AUDIO_FEATURE_TRANSFORM_REGISTRY[configs.audio.name](configs)
        self._load_audio = load_audio

        if self.apply_spec_augment:
            self._spec_augment = SpecAugment(
                freq_mask_para=configs.augment.freq_mask_para,
                freq_mask_num=configs.augment.freq_mask_num,
                time_mask_num=configs.augment.time_mask_num,
            )
            for idx in range(self.dataset_size):
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])
                self.augments.append(self.SPEC_AUGMENT)
            
        if self.apply_noise_augment:
            self._spec_augment = SpecAugment(
                freq_mask_para=configs.augment.freq_mask_para,
                freq_mask_num=configs.augment.freq_mask_num,
                time_mask_num=configs.augment.time_mask_num,
            )
            for idx in range(self.dataset_size):
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])
                self.augments.append(self.SPEC_AUGMENT)

        if self.apply_noise_augment:
            if eval(configs.augment.noise_dataset_dir) is None:
                raise ValueError("`noise_dataset_dir` should be contain audio files.")

            self._noise_injector = NoiseInjector(
                noise_dataset_dir=configs.augment.noise_dataset_dir,
                sample_rate=configs.augment.noise_sample_rate,
                noise_level=configs.augment.noise_level,
            )
            for idx in range(self.dataset_size):
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])
                self.augments.append(self.NOISE_AUGMENT)
            
        if self.apply_time_stretch_augment:
            self._time_stretch_augment = TimeStretchAugment(
                min_rate=configs.time_stretch_min_rate,
                max_rate=configs.time_stretch_max_rate,
            )
            for idx in range(self.dataset_size):
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])
                self.augments.append(self.TIME_AUGMENT)

        if self.apply_joining_augment:
            self._joining_augment = JoiningAugment()
            for idx in range(self.dataset_size):
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])
                self.augments.append(self.AUDIO_JOINING)

        self.total_size = len(self.audio_paths)

        tmp = list(zip(self.audio_paths, self.transcripts, self.augments))
        random.shuffle(tmp)

        for i, x in enumerate(tmp):
            self.audio_paths[i] = x[0]
            self.transcripts[i] = x[1] 
            self.augments[i] = x[2]

    def _parse_audio(self, audio_path: str, augment: int = None, joining_idx: int = 0) -> Tensor:
        signal = self._load_audio(audio_path, sample_rate=self.sample_rate, del_silence=self.del_silence)

        if signal is None:
            logger.warning(f"{audio_path} is not Valid!")
            return torch.zeros(1000, self.num_mels)
        
        if augment == self.AUDIO_JOINING:
            joining_singal = self._load_audio(self.audio_paths[joining_idx], sample_rate=self.sample_rate)
            signal = self._joining_augment((signal, joining_singal))

        if augment == self.TIME_AUGMENT:
            signal = self._time_stretch_augment(signal)

        if augment == self.NOISE_AUGMENT:
            signal = self._noise_injector(signal)

        feature = self.transforms(signal)

        feature -= feature.mean()
        feature /= np.std(feature)

        feature = torch.FloatTensor(feature).transpose(0, 1)

        if augment == self.SPEC_AUGMENT:
            feature = self._spec_augment(feature)

        return feature

    def _parse_transcript(self, transcript: str) -> list:
        tokens = transcript.split(" ")
        transcript = list()

        transcript.append(int(self.sos_id))
        for token in tokens:
            transcript.append(int(token)) # transcript should be like "id id id " foramt instead of original raw text
        transcript.append(int(self.eos_id))

        return transcript

    def __getitem__(self, idx: int):
        """Provides pair of audio & transcript"""
        audio_path = os.path.join(self.dataset_path, self.audio_paths[idx])
        
        if self.augments[idx] == self.AUDIO_JOINING:
            joining_idx = random.randint(0, self.total_size)
            feature = self._parse_audio(audio_path, self.augments[idx], joining_idx)
            transcript = self._parse_transcript(f"{self.transcripts[idx]} {self.transcripts[joining_idx]}")
            
        else:
            feature = self._parse_audio(audio_path, self.augments[idx])
            transcript = self._parse_transcript(self.transcripts[idx])
        
        return feature, transcript

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)